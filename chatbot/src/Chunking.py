#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# ----------------------------
# Regex patterns
# ----------------------------

ROMAN_ARTICLE_NUMBER = r"[IVXLCDM]+"
ARTICLE_NUMBER = r"\d+"
ARTICLE_LINE_RE = re.compile(rf"(?mi)^\s*({ARTICLE_NUMBER})\.\s*člen\s*$")
ARTICLE_INLINE_RE = re.compile(rf"(?mi)^\s*({ARTICLE_NUMBER})\.\s*člen\b")
ARTICLE_FORWARD_RE = re.compile(rf"(?mi)^\s*člen\s+({ARTICLE_NUMBER})\b")
ROMAN_ARTICLE_LINE_RE = re.compile(rf"(?mi)^\s*({ROMAN_ARTICLE_NUMBER})\.\s*člen\s*$")
ROMAN_ARTICLE_INLINE_RE = re.compile(rf"(?mi)^\s*({ROMAN_ARTICLE_NUMBER})\.\s*člen\b")
PARAGRAPH_RE = re.compile(r"(?m)^\s*\((\d+)\)\s+")
LEGAL_NUMBERED_LIST_ITEM_RE = re.compile(r"(?m)^\s*(\d+[.)])\s+")
LEGAL_LETTERED_LIST_ITEM_RE = re.compile(r"(?m)^\s*([a-zčšž]\))\s+")
LEGAL_LIST_ITEM_RE = re.compile(r"(?m)^\s*((?:\d+[.)]|[a-zčšž]\)))\s+")
INTERNAL_ARTICLE_MARKER_RE = re.compile(
    rf"(?mi)^\s*(?:Article\s+(?:{ROMAN_ARTICLE_NUMBER}|\d+)|{ROMAN_ARTICLE_NUMBER}\.\s*člen(?:\s*[–-].*)?)\s*$"
)
LAW_CODE_RE = re.compile(r"\(([^()]{2,30})\)\s*$")

# Typical chapter / part headings in Slovenian laws:
# I. SPLOŠNE DOLOČBE
# II. POSEBNE DOLOČBE
# 1. poglavje
# PRVI DEL
ROMAN_CHAPTER_RE = re.compile(
    r"(?m)^\s*([IVXLCDM]+)\.\s+[A-ZČŠŽ0-9][A-ZČŠŽ0-9\s\-\–\,/]+$"
)
ARABIC_CHAPTER_RE = re.compile(
    r"(?m)^\s*(\d+)\.\s*(poglavje|del|oddelek)\b.*$",
    re.IGNORECASE,
)
WORD_CHAPTER_RE = re.compile(
    r"(?m)^\s*(PRVI|DRUGI|TRETJI|ČETRTI|PETI|ŠESTI|SEDMI|OSMI|DEVETI|DESETI|ENAJSTI|DVANAJSTI)\s+DEL\b.*$",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(r"^\s*\d+\.\s+[A-ZČŠŽ][^.;:]{1,120}$")

NOISE_PREFIXES = [
    "Opozorilo:",
    "Neuradno prečiščeno besedilo",
    "Besedilo osnovnega predpisa",
    "ZAKON",
    "UREDBA",
    "PRAVILNIK",
]


# ----------------------------
# Domain configuration
# ----------------------------

IMPORTANT_LAW_CODES = {
    # tax core
    "ZDoh-2",
    "ZDavP-2",
    "ZDDPO-2",
    "ZDDV-1",
    # investment / finance
    "ZTFI-1",
    "ZISDU-3",
    "ZBan-3",
    "ZGD-1",
    # ...
}

IMPORTANT_TITLE_KEYWORDS = [
    "dohodnina",
    "davčni postopek",
    "davek na dodano vrednost",
    "davek od dohodkov pravnih oseb",
    "trg finančnih instrumentov",
    "investicijski sklad",
    "upravljavsk",
    "bank",
    "gospodarske družbe",
    "kapital",
    "vrednostni papir",
]

IMPORTANT_TEXT_KEYWORDS = [
    "dohodnina",
    "davčni zavezanec",
    "davčna osnova",
    "davčna stopnja",
    "akontacija dohodnine",
    "dohodek iz kapitala",
    "kapitalski dobiček",
    "dobiček iz kapitala",
    "dividenda",
    "dividende",
    "obresti",
    "odsvojitev vrednostnih papirjev",
    "delnice",
    "obveznice",
    "investicijski sklad",
    "finančni instrument",
    "vrednostni papir",
    "ddv",
    "davek",
    "davčna napoved",
    "davčni organ",
    "rezident",
    "nerezident",
    "pravna oseba",
    "borzn",
]

DOMAIN_TAG_RULES = {
    "tax": [
        "dohodnina",
        "davek",
        "davčni",
        "ddv",
        "davčna osnova",
        "davčna stopnja",
        "davčna napoved",
    ],
    "income_tax": [
        "dohodnina",
        "akontacija dohodnine",
    ],
    "capital_gains": [
        "kapitalski dobiček",
        "dobiček iz kapitala",
        "odsvojitev",
        "vrednostni papir",
    ],
    "dividends": [
        "dividend",
    ],
    "interest": [
        "obrest",
    ],
    "investment": [
        "finančni instrument",
        "investicijski sklad",
        "delnice",
        "obveznice",
        "vrednostni papir",
        "upravljavska družba",
    ],
    "corporate_tax": [
        "pravna oseba",
        "davek od dohodkov pravnih oseb",
    ],
    "vat": [
        "ddv",
        "davek na dodano vrednost",
    ],
    "procedure": [
        "davčni organ",
        "davčna napoved",
        "rok",
        "postopek",
    ],
}


# ----------------------------
# Dataclasses
# ----------------------------

@dataclass
class RawRecord:
    id: Optional[int]
    naziv: str
    mopedId: Optional[str]
    eva: Optional[str]
    epa: Optional[str]
    sop: Optional[str]
    text: str


@dataclass
class Chunk:
    chunk_id: str
    record_id: Optional[int]
    law_title: str
    law_code: Optional[str]
    mopedId: Optional[str]
    eva: Optional[str]
    epa: Optional[str]
    sop: Optional[str]
    chapter: Optional[str]
    article_number: Optional[str]
    article_label: Optional[str]
    article_heading: Optional[str]
    paragraph_number: Optional[str]
    chunk_part: Optional[str]
    domain_tags: List[str]
    source_type: str
    text: str
    formatted_text: str


# ----------------------------
# Helpers
# ----------------------------

def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u00a0 \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_law_code(title: str) -> Optional[str]:
    m = LAW_CODE_RE.search(title.strip())
    if not m:
        return None
    code = m.group(1).strip()
    return code or None


def keyword_hits(text: str, keywords: List[str]) -> int:
    lowered = text.lower()
    return sum(1 for kw in keywords if kw.lower() in lowered)


def is_relevant_record(
    title: str,
    text: str,
    strict_whitelist: bool = False,
    min_keyword_matches: int = 2,
) -> Tuple[bool, str]:
    law_code = extract_law_code(title or "")

    if law_code in IMPORTANT_LAW_CODES:
        return True, "whitelist"

    if strict_whitelist:
        return False, "not_in_whitelist"

    title_hits = keyword_hits(title, IMPORTANT_TITLE_KEYWORDS)
    text_hits = keyword_hits(text, IMPORTANT_TEXT_KEYWORDS)

    if title_hits >= 1:
        return True, "title_keyword"
    if text_hits >= min_keyword_matches:
        return True, "text_keywords"

    return False, "irrelevant"


def strip_leading_noise(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned: List[str] = []
    started_articles = False

    for line in lines:
        if ARTICLE_INLINE_RE.match(line) or ARTICLE_FORWARD_RE.match(line):
            started_articles = True

        if not started_articles:
            if any(line.startswith(prefix) for prefix in NOISE_PREFIXES):
                continue
            if line in {"", "ZAKON", "UREDBA", "PRAVILNIK"}:
                continue

        cleaned.append(line)

    return normalize_whitespace("\n".join(cleaned))


def detect_chapter_positions(text: str) -> List[Tuple[int, str]]:
    matches: List[Tuple[int, str]] = []
    for rx in (ROMAN_CHAPTER_RE, ARABIC_CHAPTER_RE, WORD_CHAPTER_RE):
        for m in rx.finditer(text):
            matches.append((m.start(), m.group(0).strip()))
    matches.sort(key=lambda x: x[0])

    deduped: List[Tuple[int, str]] = []
    seen = set()
    for pos, label in matches:
        key = (pos, label)
        if key not in seen:
            deduped.append((pos, label))
            seen.add(key)
    return deduped


def is_structural_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return any(
        rx.fullmatch(line)
        for rx in (ROMAN_CHAPTER_RE, ARABIC_CHAPTER_RE, WORD_CHAPTER_RE, NUMBERED_HEADING_RE)
    )


def assign_chapter(article_start: int, chapter_positions: List[Tuple[int, str]]) -> Optional[str]:
    current = None
    for pos, chapter in chapter_positions:
        if pos <= article_start:
            current = chapter
        else:
            break
    return current


def find_article_matches(text: str) -> List[re.Match]:
    primary_matches = (
        list(ARTICLE_LINE_RE.finditer(text))
        + list(ARTICLE_INLINE_RE.finditer(text))
        + list(ARTICLE_FORWARD_RE.finditer(text))
    )
    matches = primary_matches or (
        list(ROMAN_ARTICLE_LINE_RE.finditer(text))
        + list(ROMAN_ARTICLE_INLINE_RE.finditer(text))
    )
    matches.sort(key=lambda m: m.start())

    deduped: List[re.Match] = []
    seen_starts = set()
    for match in matches:
        if match.start() in seen_starts:
            continue
        deduped.append(match)
        seen_starts.add(match.start())
    return deduped


def parse_articles(text: str) -> List[dict]:
    matches = find_article_matches(text)
    articles: List[dict] = []
    if not matches:
        return articles

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()

        if not block:
            continue

        article_number = match.group(1)
        first_line = block.splitlines()[0].strip()
        if first_line.lower().startswith("člen "):
            article_label = f"Člen {article_number}"
            body_block = block if len(first_line.split()) > 2 else text[match.end():end].strip()
        else:
            article_label = f"{article_number}. člen"
            body_block = text[match.end():end].strip()

        article_heading = None
        body_lines = [ln.strip() for ln in body_block.splitlines() if ln.strip()]
        body_start_idx = 0
        if body_lines and re.fullmatch(r"\(.+\)", body_lines[0]):
            article_heading = body_lines[0][1:-1].strip()
            body_start_idx = 1

        body_lines = [
            line for line in body_lines[body_start_idx:]
            if not is_structural_heading(line)
        ]
        article_body = normalize_whitespace("\n".join(body_lines))

        articles.append(
            {
                "article_start": start,
                "article_number": article_number,
                "article_label": article_label,
                "article_heading": article_heading,
                "article_body": article_body,
            }
        )

    return articles


def split_paragraphs(article_body: str) -> List[Tuple[Optional[str], str]]:
    matches = list(PARAGRAPH_RE.finditer(article_body))
    if not matches:
        return [(None, article_body.strip())]

    parts: List[Tuple[Optional[str], str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(article_body)
        block = article_body[start:end].strip()
        parts.append((match.group(1), normalize_whitespace(block)))
    return parts


def split_legal_list_items(text: str) -> List[Tuple[Optional[str], str]]:
    matches = list(LEGAL_NUMBERED_LIST_ITEM_RE.finditer(text))
    if not matches:
        matches = list(LEGAL_LETTERED_LIST_ITEM_RE.finditer(text))
    if not matches:
        return [(None, text.strip())]

    prefix = text[:matches[0].start()].strip()
    parts: List[Tuple[Optional[str], str]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = match.group(1)
        block = normalize_whitespace(text[start:end])
        if prefix and i == 0:
            block = normalize_whitespace(f"{prefix}\n{block}")
        parts.append((label, block))
    return parts


def split_long_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """
    Fallback splitter if a paragraph is still very long.
    Splits on sentence-ish boundaries where possible, otherwise on character windows.
    """
    text = normalize_whitespace(text)
    if len(text) <= max_chars:
        return [text]

    candidate_breaks = [
        m.end()
        for m in re.finditer(r"[;:]\s+|(?<!\d)\.\s+|\n+", text)
    ]
    chunks: List[str] = []
    start = 0

    while start < len(text):
        target_end = min(start + max_chars, len(text))
        best_end = None

        for b in candidate_breaks:
            if start + max_chars // 2 <= b <= target_end:
                best_end = b

        end = best_end if best_end else target_end
        if best_end is None and end < len(text):
            whitespace = text.rfind(" ", start + max_chars // 2, target_end)
            if whitespace > start:
                end = whitespace

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end, start + 1)

    return chunks


def split_by_internal_article_markers(text: str) -> List[str]:
    matches = list(INTERNAL_ARTICLE_MARKER_RE.finditer(text))
    if len(matches) < 2:
        return [text.strip()]

    parts: List[str] = []
    prefix = text[:matches[0].start()].strip()
    if prefix:
        parts.append(prefix)

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        part = normalize_whitespace(text[start:end])
        if part:
            parts.append(part)

    return parts or [text.strip()]


def pack_text_units(units: List[str], max_chars: int) -> List[str]:
    packed: List[str] = []
    current: List[str] = []
    current_len = 0

    for unit in units:
        unit = normalize_whitespace(unit)
        if not unit:
            continue

        if len(unit) > max_chars:
            if current:
                packed.append(normalize_whitespace("\n".join(current)))
                current = []
                current_len = 0
            packed.extend(split_long_text(unit, max_chars=max_chars, overlap=150))
            continue

        next_len = current_len + len(unit) + (1 if current else 0)
        if current and next_len > max_chars:
            packed.append(normalize_whitespace("\n".join(current)))
            current = [unit]
            current_len = len(unit)
        else:
            current.append(unit)
            current_len = next_len

    if current:
        packed.append(normalize_whitespace("\n".join(current)))

    return packed


def split_article_body(article_body: str, max_chars: int) -> List[str]:
    article_body = normalize_whitespace(article_body)
    if len(article_body) <= max_chars:
        return [article_body]

    internal_article_units = split_by_internal_article_markers(article_body)
    if len(internal_article_units) > 1:
        return pack_text_units(internal_article_units, max_chars=max_chars)

    paragraph_units = [text for _, text in split_paragraphs(article_body) if text.strip()]
    if len(paragraph_units) > 1:
        return pack_text_units(paragraph_units, max_chars=max_chars)

    list_units = [text for _, text in split_legal_list_items(article_body) if text.strip()]
    if len(list_units) > 1:
        return pack_text_units(list_units, max_chars=max_chars)

    return split_long_text(article_body, max_chars=max_chars, overlap=150)


def assign_domain_tags(law_code: Optional[str], law_title: str, text: str) -> List[str]:
    haystack = f"{law_title}\n{text}".lower()
    tags = set()

    if law_code in {"ZDoh-2", "ZDavP-2", "ZDDPO-2", "ZDDV-1"}:
        tags.add("tax")
    if law_code in {"ZTFI-1", "ZISDU-3", "ZBan-3"}:
        tags.add("investment")
    if law_code == "ZDoh-2":
        tags.add("income_tax")
    if law_code == "ZDavP-2":
        tags.add("procedure")
    if law_code == "ZDDPO-2":
        tags.add("corporate_tax")
    if law_code == "ZDDV-1":
        tags.add("vat")

    for tag, needles in DOMAIN_TAG_RULES.items():
        if any(needle.lower() in haystack for needle in needles):
            tags.add(tag)

    return sorted(tags)


def make_formatted_text(
    law_title: str,
    law_code: Optional[str],
    chapter: Optional[str],
    article_label: Optional[str],
    article_heading: Optional[str],
    paragraph_number: Optional[str],
    chunk_part: Optional[str],
    text: str,
) -> str:
    parts = []

    if law_title:
        parts.append(f"Zakon: {law_title}")
    if law_code:
        parts.append(f"Oznaka: {law_code}")
    if chapter:
        parts.append(f"Poglavje: {chapter}")
    if article_label:
        parts.append(f"Člen: {article_label}")
    if article_heading:
        parts.append(f"Naslov člena: {article_heading}")
    if paragraph_number:
        if paragraph_number.endswith((")", ".")):
            parts.append(f"Odstavek/točka: {paragraph_number}")
        else:
            parts.append(f"Odstavek: ({paragraph_number})")
    if chunk_part:
        parts.append(f"Del člena: {chunk_part}")

    parts.append(f"Besedilo: {text}")
    return " | ".join(parts)


def make_chunk_id(
    moped_id: Optional[str],
    article_number: Optional[str],
    paragraph_number: Optional[str],
    subchunk_idx: Optional[int] = None,
) -> str:
    base = moped_id or "NO_MOPED"
    a = article_number or "DOC"
    p = paragraph_number or "FULL"
    cid = f"{base}_A{a}_P{p}"
    if subchunk_idx is not None:
        cid += f"_S{subchunk_idx}"
    return cid


def parse_record_to_chunks(
    record: RawRecord,
    max_article_chars: int,
    keep_nonparsed: bool = False,
) -> List[Chunk]:

    clean_text = strip_leading_noise(record.text)
    chapter_positions = detect_chapter_positions(clean_text)
    articles = parse_articles(clean_text)

    law_code = extract_law_code(record.naziv)

    # --- FALLBACK: NO ARTICLES ---
    if not articles:
        if not keep_nonparsed:
            return []

        fallback_text = normalize_whitespace(clean_text).strip()
        if not fallback_text:
            return []

        # enforce max length even here
        subchunks = split_long_text(
            fallback_text, max_chars=max_article_chars, overlap=150
        )

        output = []
        for idx, sub in enumerate(subchunks, start=1):
            sub = sub.strip()
            if not sub:
                continue

            tags = assign_domain_tags(law_code, record.naziv, sub)

            output.append(
                Chunk(
                    chunk_id=make_chunk_id(record.mopedId, None, None, subchunk_idx=idx),
                    record_id=record.id,
                    law_title=record.naziv,
                    law_code=law_code,
                    mopedId=record.mopedId,
                    eva=record.eva,
                    epa=record.epa,
                    sop=record.sop,
                    chapter=None,
                    article_number=None,
                    article_label=None,
                    article_heading=None,
                    paragraph_number=None,
                    chunk_part=f"{idx}/{len(subchunks)}" if len(subchunks) > 1 else None,
                    domain_tags=tags,
                    source_type="COLESLAW",
                    text=sub,
                    formatted_text=make_formatted_text(
                        law_title=record.naziv,
                        law_code=law_code,
                        chapter=None,
                        article_label=None,
                        article_heading=None,
                        paragraph_number=None,
                        chunk_part=f"{idx}/{len(subchunks)}" if len(subchunks) > 1 else None,
                        text=sub,
                    ),
                )
            )

        return output

    # --- NORMAL FLOW ---
    output: List[Chunk] = []

    for article in articles:
        chapter = assign_chapter(article["article_start"], chapter_positions)

        article_body = article["article_body"]
        article_number = article["article_number"]
        article_label = article["article_label"]
        article_heading = article["article_heading"]

        if not article_body:
            continue

        article_parts = split_article_body(article_body, max_article_chars)
        for idx, chunk_text in enumerate(article_parts, start=1):
            tags = assign_domain_tags(law_code, record.naziv, chunk_text)
            chunk_part = f"{idx}/{len(article_parts)}" if len(article_parts) > 1 else None
            output.append(
                Chunk(
                    chunk_id=make_chunk_id(
                        record.mopedId,
                        article_number,
                        None,
                        subchunk_idx=idx if len(article_parts) > 1 else None,
                    ),
                    record_id=record.id,
                    law_title=record.naziv,
                    law_code=law_code,
                    mopedId=record.mopedId,
                    eva=record.eva,
                    epa=record.epa,
                    sop=record.sop,
                    chapter=chapter,
                    article_number=article_number,
                    article_label=article_label,
                    article_heading=article_heading,
                    paragraph_number=None,
                    chunk_part=chunk_part,
                    domain_tags=tags,
                    source_type="COLESLAW",
                    text=chunk_text,
                    formatted_text=make_formatted_text(
                        record.naziv,
                        law_code,
                        chapter,
                        article_label,
                        article_heading,
                        None,
                        chunk_part,
                        chunk_text,
                    ),
                ),
            )

    return output


# ----------------------------
# CLI / main
# ----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse and chunk COLESLAW JSONL.")
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Path to output chunks JSONL")
    parser.add_argument("--report", type=Path, default=None, help="Optional output report JSON")
    parser.add_argument(
        "--strict-whitelist",
        action="store_true",
        help="Keep only laws in IMPORTANT_LAW_CODES",
    )
    parser.add_argument(
        "--max-article-chars",
        type=int,
        default=1500,
        help="Maximum chars per chunk; oversized articles are split while keeping article metadata.",
    )
    parser.add_argument(
        "--min-keyword-matches",
        type=int,
        default=2,
        help="For non-whitelist laws, require at least this many body keyword hits",
    )
    parser.add_argument(
        "--keep-nonparsed",
        action="store_true",
        help="If no articles are detected, keep a fallback document-level chunk",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    counters = Counter()
    reason_counters = Counter()
    tag_counters = Counter()
    law_chunk_counts = Counter()
    law_record_counts = Counter()

    output_rows: List[dict] = []

    for obj in read_jsonl(args.input):
        record = RawRecord(
            id=obj.get("id"),
            naziv=obj.get("naziv", ""),
            mopedId=obj.get("mopedId"),
            eva=obj.get("eva"),
            epa=obj.get("epa"),
            sop=obj.get("sop"),
            text=obj.get("text", "") or "",
        )

        counters["records_total"] += 1

        relevant, reason = is_relevant_record(
            record.naziv,
            record.text,
            strict_whitelist=args.strict_whitelist,
            min_keyword_matches=args.min_keyword_matches,
        )

        if not relevant:
            counters["records_skipped"] += 1
            reason_counters[reason] += 1
            continue

        counters["records_kept"] += 1
        reason_counters[reason] += 1

        law_code = extract_law_code(record.naziv) or "UNKNOWN"
        law_record_counts[law_code] += 1

        chunks = parse_record_to_chunks(
            record,
            max_article_chars=args.max_article_chars,
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
                tag_counters[tag] += 1
            output_rows.append(asdict(chunk))

    write_jsonl(args.output, output_rows)

    report = {
        "input_file": str(args.input),
        "output_file": str(args.output),
        "records_total": counters["records_total"],
        "records_kept": counters["records_kept"],
        "records_skipped": counters["records_skipped"],
        "records_no_chunks": counters["records_no_chunks"],
        "records_with_chunks": counters["records_with_chunks"],
        "chunks_total": counters["chunks_total"],
        "filter_reasons": dict(reason_counters),
        "top_laws_by_chunks": law_chunk_counts.most_common(30),
        "top_laws_by_records": law_record_counts.most_common(30),
        "domain_tag_counts": dict(tag_counters),
        "config": {
            "strict_whitelist": args.strict_whitelist,
            "max_article_chars": args.max_article_chars,
            "min_keyword_matches": args.min_keyword_matches,
            "keep_nonparsed": args.keep_nonparsed,
        },
    }

    if args.report:
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
