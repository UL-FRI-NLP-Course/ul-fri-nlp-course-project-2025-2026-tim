import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download


class LLM:
    def __init__(self, settings):
        self.settings = settings
        self.models_root = Path(settings.model_dir_path).resolve()
        
        if self.settings.nn_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

        if not self.models_root.exists():
            raise ValueError(f"Model directory does not exist: {self.models_root}")

        model_name = self.settings.LLM_model
        path = (self.models_root / model_name).resolve()

        if not str(path).startswith(str(self.models_root)):
            raise ValueError("Model path escapes /models directory")

        self.model_path = path

        self.cache_dir = self.models_root / "hf_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)

        self.device = torch.device(self.settings.nn_device)
        self.tokenizer = None
        self.model = None
        
    def download_if_missing(self):
        if self.model_path.exists() and (self.model_path / "config.json").exists():
            return

        print(f"Downloading model '{self.settings.LLM_model}' into {self.model_path}...")

        self.model_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.settings.LLM_model,
            local_dir=self.model_path,
            local_dir_use_symlinks=False
        )

        print("Download complete.")

    def load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found in /models: {self.model_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = None) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first!")

        settings = self.settings

        if max_new_tokens is None:
            max_new_tokens = settings.llm_max_new_tokens

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=settings.llm_do_sample,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                top_k=settings.llm_top_k_sampling,
                repetition_penalty=settings.llm_repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)