# RAG Evaluation Framework

This folder contains a two-step evaluation pipeline for the tax-law RAG chatbot.

## Structure

```text
evaluation/
  data/gold_eval.jsonl
  prompts/chatbot_system_prompt.yaml
  prompts/judge_prompt.yaml
  scripts/generate_predictions.py
  scripts/evaluate_predictions.py
  outputs/
```

## Data

`evaluation/data/gold_eval.jsonl` contains one JSON object per test case:

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "gold_sources": [...]
}
```

## Step 1: Generate Predictions

`generate_predictions.py` reads the gold questions, runs the chatbot with RAG retrieval, and writes answers to `evaluation/outputs/predictions/predictions.jsonl`.

Shared secrets and API credentials live in the repo-root `.env` file. See `.env.example` for the expected variables.

From the `chatbot/` directory, you can also use the convenience runner:

```bash
bash run_predictions.sh
```

```bash
python evaluation/scripts/generate_predictions.py \
  --gold evaluation/data/gold_eval.jsonl \
  --prompt evaluation/prompts/chatbot_system_prompt.yaml \
  --output evaluation/outputs/predictions/predictions.jsonl \
  --chatbot-config chatbot/configs/config.yaml \
  --llm-base-url http://localhost:8000 \
  --llm-model meta-llama/Llama-3.1-8B-Instruct
```

If `--llm-base-url` and `--llm-model` are omitted, the script tries to read them from `chatbot/configs/server_boot_config.yaml`.

Use `--limit N` for a quick smoke test.

## Step 2: Judge Predictions

`evaluate_predictions.py` asks an LLM judge to compare each model answer against the gold answer and gold sources.

If `OPENAI_API_KEY` is set in the repo-root `.env` file, the script uses OpenAI directly by default. Set `OPENAI_JUDGE_MODEL` to choose the hosted judge model; the default is `gpt-5-nano`. If `OPENAI_API_KEY` is not set, the script falls back to the local server boot config or any explicit `--judge-base-url` / `--judge-model` flags you pass.

From the `chatbot/` directory, you can also use the convenience runner:

```bash
bash run_eval.sh
```

```bash
python evaluation/scripts/evaluate_predictions.py \
  --input evaluation/outputs/predictions/predictions.jsonl \
  --output evaluation/outputs/evaluations/judgments.jsonl \
  --summary evaluation/outputs/evaluations/judgment_summary.json \
  --judge-prompt evaluation/prompts/judge_prompt.yaml \
  --judge-base-url http://localhost:8000 \
  --judge-model meta-llama/Llama-3.1-8B-Instruct
```

The judge writes per-example scores:

```json
{
  "correctness_score": 0,
  "completeness_score": 0,
  "faithfulness_score": 0,
  "citation_score": 0,
  "overall_score": 0,
  "passed": false,
  "rationale": "..."
}
```

The summary file includes averages and pass rate.

## Pipeline

```text
gold_eval.jsonl
   -> generate_predictions.py
   -> outputs/predictions/predictions.jsonl
   -> evaluate_predictions.py
   -> outputs/evaluations/judgments.jsonl + judgment_summary.json
```

The two shell entrypoints in `chatbot/` follow the same split:

- `run_predictions.sh` generates `outputs/predictions/predictions.jsonl`
- `run_eval.sh` evaluates existing predictions and writes `outputs/evaluations/judgments.jsonl` plus `outputs/evaluations/judgment_summary.json`
