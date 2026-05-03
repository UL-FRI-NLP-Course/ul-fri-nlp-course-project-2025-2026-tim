import json
import requests

INPUT = "eval/data/gold_eval.jsonl"
OUTPUT = "eval/outputs/predictions.jsonl"

API_URL = "http://localhost:8000/ask"


def ask_rag(question):
    """
    Kle vpraši LLM za prediction.
    """
    res = requests.post(API_URL, json={"question": question})
    return res.json()


with open(INPUT, encoding="utf-8") as f_in, open(
    OUTPUT, "w", encoding="utf-8"
) as f_out:

    for line in f_in:
        item = json.loads(line)

        prediction = ask_rag(item["question"])  # Question from gold_eval.jsonl

        result = {
            "id": item["id"],
            "question": item["question"],
            "gold_answer": item["gold_answer"],
            "gold_sources": item["gold_sources"],
            "prediction": prediction,
        }

        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
