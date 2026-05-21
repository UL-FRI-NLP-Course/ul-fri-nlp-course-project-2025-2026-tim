# NLP Course Project: Investment & Tax Law Chatbot

A domain-specific LLM chatbot for answering questions about Slovenian tax and investment law.

---

## Enviroment Setup

All scripts and source code are in the `chatbot` directory:

```bash
cd chatbot
```

Create the container, overlay file, and install dependencies (installation takes a long time):

```bash
bash create_environment.sh
```

Copy the example enviroment file into a new `.env`

```bash
cp .env.example .env
```

Configure the enviroment file:

```bash
# Copy this file to `.env` and fill in the values you need locally.
# `LLM_MODELS_DIR` is the path where all models and chunk embeddings will be stored.
# `HUGGINGFACE_HUB_TOKEN` is needed for downloading gated Hugging Face models.
# `OPENAI_API_KEY` enables hosted OpenAI judging.
# `OPENAI_JUDGE_MODEL` defaults to `gpt-5-nano`.
# `OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`.

LLM_MODELS_DIR=/d/hpc/projects/onj_fri/{your directory}
HUGGINGFACE_HUB_TOKEN=hf_...
OPENAI_API_KEY=
OPENAI_JUDGE_MODEL=gpt-5-nano
OPENAI_BASE_URL=https://api.openai.com/v1
```
> We strongly recommend you create a new directory in the `onj_fri` shared directory.

> If you do not have a HF Hub token, generate one. `(https://huggingface.co/docs/hub/en/security-tokens)`.

> If you are not running any evaluations you can leave the other 3 variables blank.
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

Run:
```bash
   bash download_processed_chunks.sh
```

These chunks were embedded with the `BAAI/bge-m3` model, so make sure it matches in `configs/config.yaml`

---


## Running the Chatbot

**Step 1** - Start the LLM server (queued as an HPC job):
```bash
bash run_llm_server.sh
```
Server connection details are written automatically to `configs/server_boot_config.yaml` on boot. The client should automatically load them on startup.

**Step 2** - Start the chatbot client:
```bash
bash run_chatbot_client.sh
```

> The client job **must be interactive** - it provides the terminal interface for chatting with the model.

---

## Evaluation

If `OPENAI_API_KEY` is present in `.env`, `evaluation/scripts/evaluate_predictions.py` uses OpenAI directly by default. Set `OPENAI_JUDGE_MODEL` in `.env` to pick the hosted judge model; the default is `gpt-5-nano`.

From the `chatbot/` directory, you can run:

```bash
bash run_predictions.sh
```

to generate `evaluation/outputs/predictions/predictions.jsonl`, and:

```bash
bash run_eval.sh
```

to evaluate those predictions and write `evaluation/outputs/evaluations/judgments.jsonl` plus `evaluation/outputs/evaluations/judgment_summary.json`.

The detailed evaluation pipeline is documented in `evaluation/scripts/Readme.md`.
