import threading
from pathlib import Path
from huggingface_hub import snapshot_download

from vllm import LLM, SamplingParams


class LLMHandler:
    def __init__(self, settings):
        self.settings = settings

        self.model = None
        self.model_path = None

        self.sessions = {}
        self.lock = threading.Lock()

    def download_if_missing(self, model_path):
        model_path = Path(model_path).resolve()

        root = Path(self.settings.model_dir_path).resolve()
        if not str(model_path).startswith(str(root)):
            raise ValueError("Model path escapes /models directory")

        if model_path.exists() and (model_path / "config.json").exists():
            return

        model_path.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=self.settings.LLM_model,
            local_dir=model_path,
            local_dir_use_symlinks=False
        )
        
    def load_model(self, model_path):
        self.model_path = str(Path(model_path).resolve())

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=1,
            trust_remote_code=True
        )


    def get_session(self, session_id):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            return self.sessions[session_id]

    def append_to_session(self, session_id, role, text):
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append((role, text))




    def build_prompt(self, session_id, user_prompt):
        with self.lock:
            history = list(self.sessions.get(session_id, []))

        prompt = ""
        for role, text in history:
            prompt += f"{role}: {text}\n"

        prompt += f"User: {user_prompt}\nAssistant:"
        return prompt
    
    

    def generate_stream(self, prompt):
        sampling_params = SamplingParams(
            max_tokens=self.settings.llm_max_new_tokens,
            temperature=self.settings.llm_temperature,
            top_p=self.settings.llm_top_p,
            top_k=self.settings.llm_top_k_sampling,
            repetition_penalty=self.settings.llm_repetition_penalty,
        )

        try:
            outputs = self.model.generate(
                [prompt],
                sampling_params
            )
        except Exception as e:
            yield f"\n[ERROR:init] {type(e).__name__}: {e}\n"
            return

        try:
            text = outputs[0].outputs[0].text

            for char in text:
                yield char

        except Exception as e:
            yield f"\n[ERROR:stream] {type(e).__name__}: {e}\n"

        finally:
            yield "\n"
            
            
    