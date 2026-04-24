import socket
from fastapi import FastAPI, Request, APIRouter
from fastapi.responses import StreamingResponse
import uvicorn
import sys
from datetime import datetime
from pathlib import Path
import yaml

from LLMHandler import LLMHandler
from ChatbotSettings import load_settings

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

CONFIG_FILE_PATH = "/workspace/config.yaml"
SETTINGS = load_settings(CONFIG_FILE_PATH)
MODEL_PATH = f"{SETTINGS.model_dir_path}/{SETTINGS.LLM_model}"

PORT = get_free_port()
API_PREFIX = "/NLP_TIM"

app = FastAPI()
router = APIRouter(prefix=API_PREFIX)

handler = None

@app.on_event("startup")
async def startup_event():
    global handler

    handler = LLMHandler(SETTINGS)

    handler.download_if_missing(MODEL_PATH)
    handler.load_model(MODEL_PATH)

    hostname = socket.gethostname()
    base_url = f"http://{hostname}:{PORT}"
    full_url = f"{base_url}{API_PREFIX}"

    print(f"LLM Server running on: {hostname}")
    print(f"Base URL: {base_url}")
    print(f"API Prefix: {API_PREFIX}")
    print(f"Full API URL: {full_url}")

    config_dir = Path(SETTINGS.chatbot_dir_path).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / SETTINGS.server_boot_file_name

    if config_file.exists():
        config_file.unlink()

    config_data = {
        "hostname": hostname,
        "base_url": base_url,
        "api_prefix": API_PREFIX,
        "full_url": full_url,
        "port": PORT,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    with open(config_file, "w") as f:
        yaml.dump(config_data, f)

    print(f"Server config written to: {config_file}")





@router.get("/")
def health():
    return {"status": "ok"}




@router.post("/generate_stream")
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




@router.post("/shutdown")
async def shutdown():
    sys.exit(0)





app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        reload=False
    )