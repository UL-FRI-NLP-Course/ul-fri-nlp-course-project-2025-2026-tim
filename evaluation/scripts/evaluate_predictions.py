from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean
from typing import Any

from openai import OpenAI
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    REPO_ROOT / "evaluation" / "outputs" / "predictions" / "predictions.jsonl"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "evaluation" / "outputs" / "evaluations" / "judgments.jsonl"
)
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT
    / "evaluation"
    / "outputs"
    / "evaluations"
    / "judgment_summary.json"
)
DEFAULT_PROMPT_PATH = REPO_ROOT / "evaluation" / "prompts" / "judge_prompt.yaml"
DEFAULT_SERVER_BOOT_CONFIG = REPO_ROOT / "chatbot" / "configs" / "server_boot_config.yaml"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5"

BOOLEAN_JUDGMENT_FIELDS = [
    "legal_correct",
    "citation_correct",
    "hallucination",
    "passed",
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("export "):
                line = line[len("export ") :].strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key or key in os.environ:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate chatbot predictions against gold answers with an LLM judge."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to predictions.jsonl from generate_predictions.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Where to write per-example judgments.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Where to write aggregate metrics.",
    )
    parser.add_argument(
        "--judge-prompt",
        type=Path,
        default=DEFAULT_PROMPT_PATH,
        help="Path to the judge system prompt YAML or text file.",
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help=(
            "Judge base URL. Defaults to OPENAI_BASE_URL or "
            f"'{DEFAULT_OPENAI_BASE_URL}' when OPENAI_API_KEY is set."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "Judge model name. Defaults to OPENAI_JUDGE_MODEL or "
            f"'{DEFAULT_OPENAI_JUDGE_MODEL}' when OPENAI_API_KEY is set."
        ),
    )
    parser.add_argument(
        "--server-boot-config",
        type=Path,
        default=DEFAULT_SERVER_BOOT_CONFIG,
        help="Optional fallback boot config with base_url and model_path.",
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing an optional judge API key.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N predictions.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each judge request.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first failed judgment instead of continuing.",
    )
    return parser.parse_args()


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_prompt(prompt_path: Path) -> str:
    with open(prompt_path, "r", encoding="utf-8") as handle:
        if prompt_path.suffix.lower() in {".yaml", ".yml"}:
            prompt_yaml = yaml.safe_load(handle)
            if isinstance(prompt_yaml, dict):
                return str(prompt_yaml["text"]).strip()
            if isinstance(prompt_yaml, str):
                return prompt_yaml.strip()
            raise ValueError(
                f"Unsupported prompt format in '{prompt_path}'. "
                "Expected either a YAML object with a 'text' field or raw text."
            )

        return handle.read().strip()


def load_boot_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def normalize_openai_base_url(base_url: str) -> str:
    if base_url.endswith("/chat/completions"):
        return base_url[: -len("/chat/completions")]
    return base_url.rstrip("/")


def determine_judge_runtime(
    args: argparse.Namespace, api_key: str | None
) -> tuple[str, str]:
    base_url = args.judge_base_url
    judge_model = args.judge_model

    if base_url and judge_model:
        return normalize_openai_base_url(base_url), judge_model

    if api_key:
        base_url = base_url or os.getenv("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL
        judge_model = (
            judge_model or os.getenv("OPENAI_JUDGE_MODEL") or DEFAULT_OPENAI_JUDGE_MODEL
        )
        return normalize_openai_base_url(base_url), judge_model

    if args.server_boot_config.exists():
        boot_config = load_boot_config(args.server_boot_config)
        base_url = base_url or boot_config.get("base_url")
        judge_model = judge_model or boot_config.get("model_path")

    if not base_url or not judge_model:
        raise FileNotFoundError(
            "Could not resolve the judge LLM endpoint. Pass both --judge-base-url "
            f"and --judge-model, or provide a boot config at '{args.server_boot_config}'."
        )

    return base_url, judge_model


def compact_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for source in sources:
        compact.append(
            {
                "law_title": source.get("law_title"),
                "law_code": source.get("law_code"),
                "article_label": source.get("article_label"),
                "paragraph_number": source.get("paragraph_number"),
                "text": source.get("text"),
            }
        )
    return compact


def build_judge_input(item: dict[str, Any]) -> str:
    prediction = item.get("prediction") or {}
    payload = {
        "id": item.get("id"),
        "question": item.get("question"),
        "gold_answer": item.get("gold_answer"),
        "gold_sources": compact_sources(item.get("gold_sources") or []),
        "model_answer": prediction.get("answer"),
        "model_sources": compact_sources(prediction.get("sources") or []),
        "prediction_error": prediction.get("error"),
    }
    return (
        "Evaluate this chatbot answer. Return only JSON matching the schema.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def parse_judge_json(content: str) -> dict[str, Any]:
    content = content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])

    for field in BOOLEAN_JUDGMENT_FIELDS:
        if field not in parsed:
            raise ValueError(f"Judge response is missing '{field}'")

    parsed["legal_correct"] = coerce_bool(parsed["legal_correct"])
    parsed["citation_correct"] = coerce_bool(parsed["citation_correct"])
    parsed["hallucination"] = coerce_bool(parsed["hallucination"])
    parsed["passed"] = coerce_bool(parsed["passed"])
    parsed["rationale"] = str(parsed.get("rationale", "")).strip()
    return parsed


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def judge_item(
    item: dict[str, Any],
    judge_prompt: str,
    judge_base_url: str,
    judge_model: str,
    api_key: str | None,
    timeout: int,
) -> dict[str, Any]:
    client = OpenAI(
        # The SDK requires an API key value even when a local OpenAI-compatible
        # server ignores authentication.
        api_key=api_key or "EMPTY",
        base_url=judge_base_url,
        timeout=timeout,
    )
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": judge_prompt},
            {"role": "user", "content": build_judge_input(item)},
        ],
        top_p=1,
    )

    content = response.choices[0].message.content or ""
    judgment = parse_judge_json(content)
    judgment["raw_judge_response"] = content
    return judgment


def summarize(judgments: list[dict[str, Any]], judge_model: str) -> dict[str, Any]:
    successful = [item for item in judgments if item.get("judgment")]
    summary: dict[str, Any] = {
        "judge_model": judge_model,
        "total": len(judgments),
        "judged": len(successful),
        "failed": len(judgments) - len(successful),
    }

    if not successful:
        return summary

    for field in BOOLEAN_JUDGMENT_FIELDS:
        summary[f"{field}_rate"] = mean(
            1.0 if item["judgment"].get(field) else 0.0 for item in successful
        )
    return summary


def main():
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv(args.api_key_env)
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.summary = args.summary.resolve()
    args.judge_prompt = args.judge_prompt.resolve()
    args.server_boot_config = args.server_boot_config.resolve()

    judge_prompt = load_prompt(args.judge_prompt)
    judge_base_url, judge_model = determine_judge_runtime(args, api_key)

    examples = read_jsonl(args.input)
    if args.limit is not None:
        examples = list(examples)[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    judgments = []

    with open(args.output, "w", encoding="utf-8") as handle:
        for index, item in enumerate(examples, start=1):
            print(f"[{index}] Evaluating prediction for {item.get('id', '')}")
            try:
                judgment = judge_item(
                    item=item,
                    judge_prompt=judge_prompt,
                    judge_base_url=judge_base_url,
                    judge_model=judge_model,
                    api_key=api_key,
                    timeout=args.timeout,
                )
                result = {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "gold_answer": item.get("gold_answer"),
                    "model_answer": (item.get("prediction") or {}).get("answer"),
                    "judgment": judgment,
                    "judge_model": judge_model,
                }
            except Exception as exc:
                if args.fail_fast:
                    raise
                result = {
                    "id": item.get("id"),
                    "question": item.get("question"),
                    "gold_answer": item.get("gold_answer"),
                    "model_answer": (item.get("prediction") or {}).get("answer"),
                    "judgment": None,
                    "judge_model": judge_model,
                    "error": str(exc),
                }

            judgments.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    summary = summarize(judgments, judge_model)
    with open(args.summary, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Saved judgments to: {args.output}")
    print(f"Saved summary to: {args.summary}")


if __name__ == "__main__":
    main()
