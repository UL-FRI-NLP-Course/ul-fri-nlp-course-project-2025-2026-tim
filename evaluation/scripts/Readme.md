# RAG Evaluation Framework

This folder contains scripts to evaluate our tax-law RAG chatbot.

---

## Structure

```
eval/
  data/gold_eval.jsonl
  scripts/run_eval.py
  scripts/judge_eval.py (optional)
  outputs/
```

---

##  Data

`gold_eval.jsonl` contains test cases:

```json
{
  "id": "...",
  "question": "...",
  "gold_answer": "...",
  "gold_sources": [...]
}
```

---

##  Run Evaluation

`run_eval.py`:

* reads questions from `gold_eval.jsonl`
* sends them to the RAG model
* saves results to `outputs/predictions.jsonl`

---

##  Scoring

Two options:

### 1. LLM judge (recommended)

Use `judge_eval.py` to compare:

* question
* gold answer
* model answer
* sources

Outputs scores like:

```json
{
  "answer_score": 0-3,
  "faithfulness_score": 0-3,
  "source_score": 0/1
}
```

### 2. Manual review

Manually check answers and assign scores.

---

## Pipeline

```
gold_eval.jsonl
   ↓
run_eval.py
   ↓
predictions.jsonl
   ↓
judge_eval.py / manual review
```

---

##  Requirements

* RAG model must return JSON with:

  * `answer`
  * `sources`

```
```
