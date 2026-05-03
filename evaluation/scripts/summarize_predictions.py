from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clean(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./../outputs/predictions.jsonl"),
        help="Path to predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./../outputs/answer_comparison.txt"),
        help="Where to save gold vs predicted answers",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out:
        for item in read_jsonl(args.input):
            pred = item.get("prediction") or {}

            out.write("=" * 100 + "\n")
            out.write(f"ID: {item.get('id', '')}\n")
            out.write(f"QUESTION:\n{clean(item.get('question'))}\n\n")
            out.write(f"GOLD ANSWER:\n{clean(item.get('gold_answer'))}\n\n")
            out.write(f"MODEL ANSWER:\n{clean(pred.get('answer'))}\n\n")

    print(f"Saved answer comparison to: {args.output}")


if __name__ == "__main__":
    main()