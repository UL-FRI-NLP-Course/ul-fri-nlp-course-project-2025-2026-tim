import socket
from pathlib import Path
from datetime import datetime
import yaml
import tempfile
import shutil

from huggingface_hub import snapshot_download
import filelock

import subprocess
import sys
import os
from ChatbotSettings import load_settings


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


CONFIG_FILE_PATH = "/workspace/configs/config.yaml"
SETTINGS = load_settings(CONFIG_FILE_PATH)

MODEL_REPO = SETTINGS.LLM_model
MODEL_ROOT = Path(SETTINGS.model_dir_path).resolve()
MODEL_PATH = (MODEL_ROOT / MODEL_REPO).resolve()

#API_PREFIX = "/NLP_TIM"
PORT = get_free_port()
HOST = "0.0.0.0"


def download_if_missing(model_path: Path):
    model_path = model_path.resolve()

    if not str(model_path).startswith(str(MODEL_ROOT)):
        raise ValueError("Model path escapes /models directory")

    config_file = model_path / "config.json"

    if config_file.exists():
        return

    model_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = str(model_path) + ".lock"

    with filelock.FileLock(lock_path):
        if config_file.exists():
            return

        print(f"[INFO] Downloading model to {model_path}")
        MODEL_ROOT.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=MODEL_ROOT) as tmp_dir:
            tmp_path = Path(tmp_dir)

            snapshot_download(
                repo_id=MODEL_REPO,
                local_dir=tmp_path,
                local_dir_use_symlinks=False,
                token=os.environ.get("HUGGINGFACE_HUB_TOKEN")
            )

            if not (tmp_path / "config.json").exists():
                raise RuntimeError("Download failed: config.json missing")

            if model_path.exists():
                shutil.rmtree(model_path)

            shutil.move(str(tmp_path), str(model_path))

        print("[INFO] Model download complete")
        
        
def start_vllm_server(model_path, host, port):
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model_path),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", "1",
        "--trust-remote-code"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    return process

if __name__ == "__main__":
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    download_if_missing(MODEL_PATH)

    hostname = socket.gethostname()
    base_url = f"http://{hostname}:{PORT}"

    print(f"LLM Server running on: {hostname}")
    print(f"Base URL: {base_url}")
    print(f"Model path: {MODEL_PATH}")

    config = {
        "hostname": hostname,
        "base_url": base_url,
        "port": PORT,
        "model_path": str(MODEL_PATH),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    CHATBOT_ROOT = Path(SETTINGS.chatbot_dir_path)
    BOOT_CONFIG_NAME = SETTINGS.server_boot_file_name
    config_file = (CHATBOT_ROOT / BOOT_CONFIG_NAME).resolve()
    config_file.parent.mkdir(parents=True, exist_ok=True)

    if config_file.exists():
        config_file.unlink()

    with open(config_file, "w") as f:
        yaml.dump(config, f)

    print(f"Server config written to: {config_file}")

    process = start_vllm_server(MODEL_PATH, HOST, PORT)

    try:
        for line in process.stdout:
            print("[vLLM]", line, end="")
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server...")
        process.terminate()
        process.wait()