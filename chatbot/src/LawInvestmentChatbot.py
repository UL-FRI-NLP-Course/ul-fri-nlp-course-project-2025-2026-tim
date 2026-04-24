import requests
from pathlib import Path
import yaml
import json


class LawInvestmentChatbot:
    def __init__(self, settings):
        self.running = False
        self.settings = settings
        self.chat_history = []

        config_path = Path(settings.chatbot_dir_path) / settings.server_boot_file_name
        if not config_path.exists():
            raise FileNotFoundError(
                f"Server boot config not found: {config_path}"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        try:
            base_url = config["base_url"]
            model = config["model_path"]
        except KeyError:
            raise ValueError("Invalid server config")

        self.server_url = f"{base_url}/v1/completions"
        self.llm_model = model


    def _handle_command(self, command: str):
        if command == "/help":
            print(
                "Available commands:\n"
                "  /help  - Show this help message\n"
                "  /exit  - Exit the chatbot"
            )
        elif command == "/exit":
            self.running = False
        else:
            print(f"Unknown command: {command}. Type /help for available commands.")


    def _build_prompt(self, text):
        prompt = ""
        for role, content in self.chat_history:
            if role == "user":
                prompt += f"User: {content}\n"
            else:
                prompt += f"Assistant: {content}\n"

        prompt += f"User: {text}\nAssistant:"
        return prompt


    def _handle_question(self, text: str):
        print("Assistant:", end=" ", flush=True)
        full_response = ""

        prompt = self._build_prompt(text)

        try:
            response = requests.post(
                self.server_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": True,
                    "temperature": self.settings.llm_temperature,
                    "max_tokens": self.settings.llm_max_new_tokens,
                },
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if line.startswith("data: "):
                    data = line[len("data: "):]

                    if data.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        token = chunk["choices"][0]["text"]

                        print(token, end="", flush=True)
                        full_response += token

                    except Exception:
                        continue

        except requests.exceptions.RequestException as e:
            print(f"\n[Error] Failed to connect to LLM server: {e}")
            return

        print()

        # Update history AFTER generation (prevents duplication)
        self.chat_history.append(("user", text))
        if full_response.strip():
            self.chat_history.append(("assistant", full_response.strip()))


    def _handle_chatbot_exit(self):
        print("Exiting chatbot...")


    def run(self):
        self.running = True

        print("\n===============================\n")
        print("Law Investment Chatbot (Remote LLM)")
        print("Type /help for commands.\n")

        while self.running:
            try:
                user_input_string = input(">> ").strip()

                if not user_input_string:
                    continue

                if user_input_string.startswith("/"):
                    self._handle_command(user_input_string)
                else:
                    self._handle_question(user_input_string)

            except (EOFError, KeyboardInterrupt):
                self.running = False
                break

        self._handle_chatbot_exit()