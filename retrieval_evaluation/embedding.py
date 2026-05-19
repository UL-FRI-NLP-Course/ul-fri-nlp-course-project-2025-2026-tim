from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from pathlib import Path


def load_embedding_model(model_name, model_run_device):
    device = model_run_device if torch.cuda.is_available() and model_run_device == 'cuda' else "cpu"
    model_dir = Path("./models") / model_name.replace("/", "_")

    if model_dir.exists():
        model = SentenceTransformer(str(model_dir), device=device)
    else:
        model = SentenceTransformer(model_name, device=device)
        model.save(str(model_dir))

    return model


def load_reranking_model(reranking_model_name, model_run_device):
    device = model_run_device if torch.cuda.is_available() and model_run_device == 'cuda' else "cpu"
    model_dir = Path("./models") / reranking_model_name.replace("/", "_")

    if model_dir.exists():
        model = CrossEncoder(str(model_dir), device=device)
    else:
        model = CrossEncoder(reranking_model_name, device=device)
        model.save(str(model_dir))

    return model


def embed_string(model, string, normalize_embeddings):
    emb = model.encode(
        string,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize_embeddings,
    )
    return emb.tolist()


def rerank_candidates(reranker, query_string, candidates, rerank_return_n):
    cross_inputs = [[query_string, c["formatted_text"]] for c in candidates]
    cross_scores = reranker.predict(cross_inputs)

    enriched = []
    for chunk, cross in zip(candidates, cross_scores):
        enriched.append({**chunk, "cross_score": float(cross)})

    enriched.sort(key=lambda x: x["cross_score"], reverse=True)
    return enriched[:min(len(enriched), rerank_return_n)]