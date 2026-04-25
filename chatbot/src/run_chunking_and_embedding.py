#!/usr/bin/env python3
"""
Parse, chunk, embed, and index COLESLAW records for the chatbot RAG store.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, List

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_: object):
        return iterable

from Chunking import (
    Chunk,
    RawRecord,
    extract_law_code,
    is_relevant_record,
    parse_record_to_chunks,
    read_jsonl,
    write_jsonl,
)


def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> Any:
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )
    return embeddings.astype("float32")


def save_faiss_index(embeddings: np.ndarray, path: Path, metric: str) -> None:
    try:
        import faiss
    except ImportError as e:
        raise ImportError("FAISS is not installed. Run: pip install faiss-cpu") from e

    dim = embeddings.shape[1]
    if metric == "cosine":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be either 'cosine' or 'l2'")

    index.add(embeddings)
    faiss.write_index(index, str(path))


def embedding_text(chunk: Chunk, passage_prefix: str) -> str:
    text = chunk.formatted_text
    if passage_prefix:
        return f"{passage_prefix}{text}"
    return text


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create chunks, embeddings, and an optional FAISS index from COLESLAW JSONL."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--all-laws",
        action="store_true",
        help="Process all records instead of filtering to tax/investment-related laws.",
    )
    parser.add_argument(
        "--strict-whitelist",
        action="store_true",
        help="When filtering, keep only laws explicitly listed in Chunking.IMPORTANT_LAW_CODES.",
    )
    parser.add_argument(
        "--min-keyword-matches",
        type=int,
        default=2,
        help="Keyword hits required for non-whitelisted records when filtering.",
    )
    parser.add_argument(
        "--keep-nonparsed",
        action="store_true",
        help="Keep a fallback document-level chunk when no articles are detected.",
    )
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--model", type=str, default="intfloat/multilingual-e5-base")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Write chunks and report without creating embeddings or a FAISS index.",
    )
    parser.add_argument(
        "--passage-prefix",
        type=str,
        default="passage: ",
        help="Prefix added before embedded chunk text. Use '' to disable.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not normalize embeddings. Default is normalized embeddings.",
    )
    parser.add_argument(
        "--no-faiss",
        action="store_true",
        help="Do not create FAISS index.",
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        default="cosine",
        help="FAISS metric. Cosine uses inner product over normalized embeddings.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = args.out_dir / "chunks.jsonl"
    embeddings_path = args.out_dir / "embeddings.npy"
    faiss_path = args.out_dir / "index.faiss"
    report_path = args.out_dir / "report.json"

    counters = Counter()
    filter_reasons = Counter()
    law_record_counts = Counter()
    law_chunk_counts = Counter()
    tag_counts = Counter()
    all_chunks: List[Chunk] = []

    for obj in tqdm(read_jsonl(args.input), desc="Parsing and chunking"):
        counters["records_total"] += 1
        record = RawRecord(
            id=obj.get("id"),
            naziv=obj.get("naziv", ""),
            mopedId=obj.get("mopedId"),
            eva=obj.get("eva"),
            epa=obj.get("epa"),
            sop=obj.get("sop"),
            text=obj.get("text", "") or "",
        )

        if args.all_laws:
            relevant, reason = True, "all_laws"
        else:
            relevant, reason = is_relevant_record(
                record.naziv,
                record.text,
                strict_whitelist=args.strict_whitelist,
                min_keyword_matches=args.min_keyword_matches,
            )

        filter_reasons[reason] += 1
        if not relevant:
            counters["records_skipped"] += 1
            continue

        counters["records_kept"] += 1
        law_code = extract_law_code(record.naziv) or "UNKNOWN"
        law_record_counts[law_code] += 1

        chunks = parse_record_to_chunks(
            record,
            max_article_chars=args.max_chars,
            keep_nonparsed=args.keep_nonparsed,
        )
        if not chunks:
            counters["records_no_chunks"] += 1
            continue

        counters["records_with_chunks"] += 1
        counters["chunks_total"] += len(chunks)
        law_chunk_counts[law_code] += len(chunks)

        for chunk in chunks:
            for tag in chunk.domain_tags:
                tag_counts[tag] += 1

        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks were created. Check filtering settings or input format.")

    write_jsonl(chunks_path, (asdict(chunk) for chunk in all_chunks))

    embeddings_shape = None
    if not args.chunks_only:
        texts = [embedding_text(chunk, args.passage_prefix) for chunk in all_chunks]
        embeddings = embed_texts(
            texts=texts,
            model_name=args.model,
            batch_size=args.batch_size,
            normalize_embeddings=not args.no_normalize,
        )
        embeddings_shape = list(embeddings.shape)

        import numpy as np

        np.save(embeddings_path, embeddings)

        if not args.no_faiss:
            save_faiss_index(embeddings=embeddings, path=faiss_path, metric=args.metric)

    report = {
        "input_file": str(args.input),
        "out_dir": str(args.out_dir),
        "chunks_path": str(chunks_path),
        "embeddings_path": None if args.chunks_only else str(embeddings_path),
        "faiss_path": None if args.no_faiss or args.chunks_only else str(faiss_path),
        "model": args.model,
        "embedding_shape": embeddings_shape,
        "records_total": counters["records_total"],
        "records_kept": counters["records_kept"],
        "records_skipped": counters["records_skipped"],
        "records_no_chunks": counters["records_no_chunks"],
        "records_with_chunks": counters["records_with_chunks"],
        "chunks_total": counters["chunks_total"],
        "filter_reasons": dict(filter_reasons),
        "top_laws_by_chunks": law_chunk_counts.most_common(30),
        "top_laws_by_records": law_record_counts.most_common(30),
        "domain_tag_counts": dict(tag_counts),
        "config": {
            "all_laws": args.all_laws,
            "strict_whitelist": args.strict_whitelist,
            "max_chars": args.max_chars,
            "min_keyword_matches": args.min_keyword_matches,
            "keep_nonparsed": args.keep_nonparsed,
            "chunks_only": args.chunks_only,
            "passage_prefix": args.passage_prefix,
            "normalize_embeddings": not args.no_normalize,
            "metric": args.metric,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
