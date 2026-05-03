
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch
from pathlib import Path

class RAGHandler:

    def __init__(self, settings):
        self.settings = settings

        self.rag_folder_path = os.path.join(settings.rag_data_path)

        print('Loading RAG Data')
        chunks_path = os.path.join(self.rag_folder_path, "chunks.jsonl")
        self.rag_chunks = self._load_jsonl(chunks_path)

        emb_path = os.path.join(self.rag_folder_path, "embeddings.npy")
        self.rag_embeddings_matrix = np.load(emb_path)

        # print(f"Loaded {len(self.rag_chunks)} chunks, "
        #     f"embedding matrix shape: {self.rag_embeddings_matrix.shape}"
        # )
        
        # MODEL INIT
        device = "cuda" if torch.cuda.is_available() and settings.nn_device == "cuda" else "cpu"
        print(f"[INFO] RAG Using device: {device}")

        embedding_model_path = os.path.join(settings.model_dir_path, settings.embedding_model)
        reranking_model_path = os.path.join(settings.model_dir_path, settings.reranking_model)

        self.embedding_model = self._load_or_download_embedding_model(settings.embedding_model, embedding_model_path, device)
        self.reranking_model = self._load_or_download_cross_encoder(settings.reranking_model, reranking_model_path, device)
        
        
    
    def _load_or_download_embedding_model(self, model_name_or_path, local_path, device):
        local_path = Path(local_path)

        try:
            if local_path.exists():
                print(f"[INFO] Loading embedding model from local path: {local_path}")
                model = SentenceTransformer(str(local_path), device=device)
            else:
                print(f"[INFO] Downloading embedding model: {model_name_or_path}")
                model = SentenceTransformer(model_name_or_path, device=device)

                local_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(local_path))
                print(f"[INFO] Saved embedding model to {local_path}")

        except Exception as e:
            print(f"[WARN] Failed loading local model, retrying download: {e}")
            model = SentenceTransformer(model_name_or_path, device=device)

        return model


    def _load_or_download_cross_encoder(self, model_name_or_path, local_path, device):
        local_path = Path(local_path)

        try:
            if local_path.exists():
                print(f"[INFO] Loading reranker from local path: {local_path}")
                model = CrossEncoder(str(local_path), device=device)
            else:
                print(f"[INFO] Downloading reranker: {model_name_or_path}")
                model = CrossEncoder(model_name_or_path, device=device)

                local_path.parent.mkdir(parents=True, exist_ok=True)
                model.save(str(local_path))
                print(f"[INFO] Saved reranker to {local_path}")

        except Exception as e:
            print(f"[WARN] Failed loading local reranker, retrying download: {e}")
            model = CrossEncoder(model_name_or_path, device=device)

        return model

    def _load_jsonl(self, path):
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data


    def _retrieve_by_query_vector(self, query_vector):
        top_k = self.settings.top_k_chunks
        scores = self.rag_embeddings_matrix @ query_vector
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [self.rag_chunks[i] for i in top_idx]


    def _embed_user_text(self, text: str):
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        text = text.strip()
        if not text:
            return np.zeros(self.rag_embeddings_matrix.shape[1], dtype=np.float32)

        text = text.replace("passage:", "")[:800]
        print(text)

        embedding = self.embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.astype(np.float32)
        

    def _reorder_retrieved_chunks(self, user_text, chunks):
        top_n = self.settings.reorder_top_n_chunks

        cross_inputs = []
        for chunk in chunks:
            chunk_record_title = chunk['law_title']
            chunk_main_text = chunk['text']
            
            chunk_text = f'{chunk_record_title}: {chunk_main_text}'
            cross_inputs.append([user_text, chunk_text])
            
        cross_scores = self.reranking_model.predict(cross_inputs)
        
        final_chunk_data = []
        for chunk, cross in zip(chunks, cross_scores):
            chunk_info = {
                'record_id' : chunk['record_id'],
                'law_title' : chunk['law_title'],
                'article_label' : chunk['article_label'],
                'paragraph_number' : chunk['paragraph_number'],
                'chunk_part' : chunk.get('chunk_part'),
                'chunk_raw_text' : chunk['text'],
                'cross_score' : cross
            }
            final_chunk_data.append(chunk_info)

        final_chunk_data.sort(key=lambda x: x["cross_score"], reverse=True)

        return final_chunk_data[:min(len(final_chunk_data), top_n)]
        
        

    def build_RAG_prompt(self, user_text, chat_history):
        query_vector = self._embed_user_text(user_text)
        retrieved_chunks = self._retrieve_by_query_vector(query_vector)
        reordered_chunks = self._reorder_retrieved_chunks(user_text, retrieved_chunks)

        context_blocks = []

        for i, chunk in enumerate(reordered_chunks):
            law_title = chunk['law_title']
            article_label = chunk['article_label']
            paragraph_number = chunk.get('paragraph_number')
            chunk_part = chunk.get('chunk_part')
            text = chunk.get('chunk_raw_text')

            block_lines = [
                f"[{i+1}]",
                f"ZAKON: {law_title}",
                f"ČLEN: {article_label}",
            ]

            if paragraph_number:
                block_lines.append(f"ODSTAVEK: {paragraph_number}")
            if chunk_part:
                block_lines.append(f"DEL ČLENA: {chunk_part}")

            block_lines.append("BESEDILO:")
            block_lines.append(text.strip())

            context_blocks.append("\n".join(block_lines))

        context_str = "\n\n".join(context_blocks)

        rag_prompt = "\n".join([
            "KONTEKST (relevantni pravni viri):",
            "",
            "Spodaj so odlomki slovenske zakonodaje. Vsak blok vsebuje:",
            "- ZAKON (ime zakona)",
            "- ČLEN",
            "- ODSTAVEK (če obstaja)",
            "- DEL ČLENA (če je bil člen zaradi dolžine razdeljen)",
            "- BESEDILO",
            "",
            context_str,
            "",
            "NAVODILA ZA UPORABO KONTEKSTA:",
            "",
            "- Odgovarjaj izključno na podlagi zgornjega konteksta.",
            "- Ne uporabljaj zunanjega znanja, če ni nujno potrebno.",
            "- Vedno jasno navedi zakon in člen, na katerega se sklicuješ.",
            '- Če odgovor ni neposredno razviden iz konteksta, napiši:',
            '  "Na podlagi podanega konteksta tega ni mogoče zanesljivo določiti."',
            "- Ne izmišljuj si zakonov ali členov.",
            "- Če obstajajo možne izjeme ali posebni pogoji, jih omeni, če so razvidni iz konteksta.",
        ])

        #print(rag_prompt)
        
        return rag_prompt
