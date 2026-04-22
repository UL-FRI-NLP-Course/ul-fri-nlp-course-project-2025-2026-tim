import socket
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import sys
from datetime import datetime
from pathlib import Path
import yaml

from LLMHandler import LLMHandler
from ChatbotSettings import load_settings

app = FastAPI()
handler = None

CONFIG_FILE_PATH = "/workspace/config.yaml"
SETTINGS = load_settings(CONFIG_FILE_PATH)
MODEL_PATH = f"{SETTINGS.model_dir_path}/{SETTINGS.LLM_model}"
PORT = 8000

@app.on_event("startup")
async def startup_event():
    global handler

    handler = LLMHandler(SETTINGS)

    handler.download_if_missing(MODEL_PATH)
    handler.load_model(MODEL_PATH)

    hostname = socket.gethostname()
    url = f"http://{hostname}:{PORT}"

    print(f"LLM Server running on: {hostname}")
    print(url)


    config_dir = Path(SETTINGS.chatbot_dir_path).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / SETTINGS.server_boot_file_name

    if config_file.exists():
        config_file.unlink()

    config_data = {
        "hostname": hostname,
        "url": url,
        "port": PORT,
        "timestamp": datetime.utcnow().isoformat() + "Z"
        
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    print(f"Server config written to: {config_file}")


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/generate_stream")
async def generate_stream(request: Request):
    global handler

    data = await request.json()

    session_id = data.get("session_id", "default")
    user_input = data["text"]

    handler.append_to_session(session_id, "User", user_input)
    full_prompt = handler.build_prompt(session_id, user_input)

    def token_stream():
        output = ""
        for token in handler.generate_stream(full_prompt):
            output += token
            yield token

        handler.append_to_session(session_id, "Assistant", output)

    return StreamingResponse(token_stream(), media_type="text/plain")


@app.post("/shutdown")
async def shutdown():
    sys.exit(0)


if __name__ == "__main__":
    uvicorn.run(
        "run_llm_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )