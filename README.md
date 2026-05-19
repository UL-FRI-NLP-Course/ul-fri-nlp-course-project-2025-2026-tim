# NLP Course Project: Investment & Tax Law Chatbot

A domain-specific LLM chatbot for answering questions about Slovenian tax and investment law.

---

## Enviroment Setup

All scripts and source code are in the `chatbot` directory:

```bash
cd chatbot
```

Create the container, overlay file, and install dependencies:

```bash
bash create_environment.sh
```

---

## Configuration

Before running anything, set the `LLM_MODELS_DIR` path in the following scripts:

| Script | Purpose |
|---|---|
| `run_llm_server.sh` | LLM inference server |
| `run_chatbot_client.sh` | Chatbot terminal client |
| `run_chunking_and_embedding.sh` | Chunk and embed raw data |
| `download_processed_chunks.sh` | Download pre-processed chunks |

> **Default path:** `/d/hpc/projects/onj_fri/group-tim`
> This shared directory may not be accessible due to permissions — set it to a directory you own on the HPC filesystem.

---

## Chunking & Embedding (skip if using pre-processed chunks)

1. Download the raw dataset:
```
   https://www.clarin.si/repository/xmlui/bitstream/handle/11356/2095/COLESLAW.zip
```
2. Place `register-predpisov.jsonl` in `chatbot/data/`
3. Run the chunking and embedding pipeline and select `normal` when prompted:
```bash
   bash run_chunking_and_embedding.sh
```

---

## Downloading Pre-Processed Chunks (skip if you ran chunking)

If you want to skip chunking and embedding, download the pre-processed data instead:

1. Set `LLM_MODELS_DIR` inside `download_processed_chunks.sh`
2. Run:
```bash
   bash download_processed_chunks.sh
```

These chunks were embedded with the `BAAI/bge-m3` model, so make sure it matches in `configs/config.yaml`

---


## Running the Chatbot

**Step 1** — Start the LLM server (queued as an HPC job):
```bash
bash run_llm_server.sh
```
Server connection details are written automatically to `configs/server_boot_config.yaml` on boot. The client should automatically load them on startup.

**Step 2** — Start the chatbot client (must be run as an interactive job):
```bash
bash run_chatbot_client.sh
```

> The client job **must be interactive** — it provides the terminal interface for chatting with the model.