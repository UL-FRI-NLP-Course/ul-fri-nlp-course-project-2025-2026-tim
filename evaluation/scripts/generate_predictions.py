from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CHATBOT_ROOT = REPO_ROOT / "chatbot"
CHATBOT_SRC = CHATBOT_ROOT / "src"

if str(CHATBOT_SRC) not in sys.path:
    sys.path.insert(0, str(CHATBOT_SRC))

from ChatbotSettings import load_settings
from RAGHandler import RAGHandler


DEFAULT_GOLD_PATH = REPO_ROOT / "evaluation" / "data" / "gold_eval.jsonl"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "evaluation" / "outputs" / "predictions" / "predictions.jsonl"
)
DEFAULT_PROMPT_PATH = (
    REPO_ROOT / "evaluation" / "prompts" / "chatbot_system_prompt.yaml"
)
DEFAULT_CHATBOT_CONFIG = CHATBOT_ROOT / "configs" / "config.yaml"
DEFAULT_MODEL_CACHE = REPO_ROOT / ".cache" / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate chatbot predictions for the gold evaluation dataset."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Path to gold_eval.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write predictions.jsonl",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the evaluation system prompt YAML or text file.",
    )
    parser.add_argument(
        "--chatbot-config",
        type=Path,
        default=DEFAULT_CHATBOT_CONFIG,
        help="Path to chatbot config.yaml",
    )
    parser.add_argument(
        "--llm-base-url",
        default=None,
        help="OpenAI-compatible base URL. Example: http://localhost:8000",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Model identifier exposed by the running vLLM server.",
    )
    parser.add_argument(
        "--server-boot-config",
        type=Path,
        default=None,
        help="Optional path to the chatbot server boot config YAML.",
    )
    parser.add_argument(
        "--model-cache-dir",
        type=Path,
        default=DEFAULT_MODEL_CACHE,
        help="Local cache directory for embedding and reranking models.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate predictions for only the first N gold examples.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each LLM request.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failed prediction instead of continuing.",
    )
    return parser.parse_args()


def resolve_chat_completion_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def load_runtime_settings(config_path: Path, model_cache_dir: Path):
    settings = load_settings(str(config_path))
    settings.chatbot_dir_path = str(CHATBOT_ROOT)

    if not Path(settings.model_dir_path).exists():
        model_cache_dir.mkdir(parents=True, exist_ok=True)
        settings.model_dir_path = str(model_cache_dir)

    return settings


def validate_runtime_assets(args: argparse.Namespace, settings) -> None:
    missing_paths = []

    if not args.gold.exists():
        missing_paths.append(args.gold)

    if not args.prompt.exists():
        missing_paths.append(args.prompt)

    if not args.chatbot_config.exists():
        missing_paths.append(args.chatbot_config)

    rag_dir = Path(settings.chatbot_dir_path) / settings.rag_data_path
    chunks_path = rag_dir / "chunks.jsonl"
    embeddings_path = rag_dir / "embeddings.npy"

    if not chunks_path.exists():
        missing_paths.append(chunks_path)

    if not embeddings_path.exists():
        missing_paths.append(embeddings_path)

    if missing_paths:
        formatted_paths = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Missing required evaluation assets:\n"
            f"{formatted_paths}\n\n"
            "Generate the RAG store first with chatbot/run_chunking_and_embedding.sh, "
            "or point the chatbot config at an existing rag_data_path."
        )


def load_prompt(prompt_path: Path) -> str:
    with open(prompt_path, "r", encoding="utf-8") as handle:
        if prompt_path.suffix.lower() in {".yaml", ".yml"}:
            prompt_yaml = yaml.safe_load(handle)
            return prompt_yaml["text"].strip()

        return handle.read().strip()


def load_boot_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def determine_llm_runtime(args: argparse.Namespace, settings) -> tuple[str, str]:
    base_url = args.llm_base_url
    llm_model = args.llm_model

    if base_url and llm_model:
        return base_url, llm_model

    boot_config_path = args.server_boot_config
    if boot_config_path is None:
        boot_config_path = (
            Path(settings.chatbot_dir_path) / settings.server_boot_file_name
        )

    if boot_config_path.exists():
        boot_config = load_boot_config(boot_config_path)
        base_url = base_url or boot_config.get("base_url")
        llm_model = llm_model or boot_config.get("model_path")

    if not base_url or not llm_model:
        raise FileNotFoundError(
            "Could not resolve the LLM endpoint. Start the chatbot LLM server first, "
            f"or pass both --llm-base-url and --llm-model. Expected boot config at "
            f"'{boot_config_path}'."
        )

    return base_url, llm_model


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)

def normalize_sources(retrieved_sources: list[dict]) -> list[dict]:
    normalized = []
    for rank, chunk in enumerate(retrieved_sources, start=1):
        normalized.append(
            {
                "rank": rank,
                "record_id": chunk.get("record_id"),
                "law_title": chunk.get("law_title"),
                "law_code": chunk.get("law_code"),
                "mopedId": chunk.get("mopedId"),
                "article_number": chunk.get("article_number"),
                "article_label": chunk.get("article_label"),
                "paragraph_number": chunk.get("paragraph_number"),
                "source_type": chunk.get("source_type"),
                "chapter": chunk.get("chapter"),
                "article_heading": chunk.get("article_heading"),
                "text": chunk.get("text"),
                "domain_tags": chunk.get("domain_tags"),
                "cross_score": chunk.get("cross_score"),
            }
        )
    return normalized


def ask_chatbot(
    question: str,
    rag_handler: RAGHandler,
    system_prompt: str,
    llm_base_url: str,
    llm_model: str,
    settings,
    timeout: int,
) -> dict:
    rag_prompt, sources = rag_handler.build_rag_context(question)

    response = requests.post(
        resolve_chat_completion_url(llm_base_url),
        json={
            "model": llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": rag_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": settings.llm_temperature,
            "top_p": settings.llm_top_p,
            "max_tokens": settings.llm_max_new_tokens,
            "presence_penalty": settings.llm_presence_penalty,
            "frequency_penalty": settings.llm_frequency_penalty,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    answer = payload["choices"][0]["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": normalize_sources(sources),
        "model": llm_model,
    }


def main():
    args = parse_args()

    args.gold = args.gold.resolve()
    args.output = args.output.resolve()
    args.prompt = args.prompt.resolve()
    args.chatbot_config = args.chatbot_config.resolve()
    if args.server_boot_config is not None:
        args.server_boot_config = args.server_boot_config.resolve()

    settings = load_runtime_settings(
        args.chatbot_config, args.model_cache_dir.resolve()
    )
    validate_runtime_assets(args, settings)
    system_prompt = load_prompt(args.prompt)
    llm_base_url, llm_model = determine_llm_runtime(args, settings)

    rag_handler = RAGHandler(settings)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    examples = read_jsonl(args.gold)
    if args.limit is not None:
        examples = list(examples)[: args.limit]

    with open(args.output, "w", encoding="utf-8") as handle:
        for index, item in enumerate(examples, start=1):
            print(f"[{index}] Generating prediction for {item['id']}")

            try:
                prediction = ask_chatbot(
                    question=item["question"],
                    rag_handler=rag_handler,
                    system_prompt=system_prompt,
                    llm_base_url=llm_base_url,
                    llm_model=llm_model,
                    settings=settings,
                    timeout=args.timeout,
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                prediction = {
                    "answer": None,
                    "sources": [],
                    "model": llm_model,
                    "error": str(exc),
                }

            result = {
                "id": item["id"],
                "question": item["question"],
                "gold_answer": item["gold_answer"],
                "gold_sources": item["gold_sources"],
                "prediction": prediction,
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
