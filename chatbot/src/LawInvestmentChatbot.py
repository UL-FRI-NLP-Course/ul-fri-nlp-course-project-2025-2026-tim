import requests
from pathlib import Path
import yaml
import json

from RAGHandler import RAGHandler

class LawInvestmentChatbot:
    def __init__(self, settings):
        self.running = False
        self.settings = settings
        self.chat_history = []
        self.RAG = RAGHandler(settings)

        # LLM SERVER COMMUNICATION
        config_path = Path(settings.chatbot_dir_path) / settings.server_boot_file_name        
        if not config_path.exists():
            raise FileNotFoundError(f"Server boot config not found: {config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        try:
            base_url = config["base_url"]
            model = config["model_path"]
        except KeyError:
            raise ValueError("Invalid server config")
        self.server_url = f"{base_url}/v1/chat/completions"
        self.llm_model = model
        
        # SYSTEM PROMPT INITIALIZATION
        system_prompt_path = Path(settings.chatbot_dir_path) / settings.system_prompt_file_name
        if not system_prompt_path.exists():
            raise FileNotFoundError(f"Initial system prompt not found: {system_prompt_path}")
        with open(system_prompt_path, "r") as f:
            system_prompt = yaml.safe_load(f)
        try:
            self.system_prompt = system_prompt['text'].strip()
        except KeyError:
            raise ValueError("Invalid system prompt")


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



    def _handle_question(self, user_text_raw: str):
        print("Vprašanje:", end=" ", flush=True)
        full_response = ""

        rag_context = self.RAG.build_RAG_prompt(user_text_raw, self.chat_history)

        messages = []
        messages.append({"role": "system", "content": self.system_prompt})
        for role, content in self.chat_history:
            messages.append({"role": role, "content": content})
        messages.append({"role": "system", "content": rag_context})
        messages.append({"role": "user", "content": user_text_raw})

        try:
            response = requests.post(
                self.server_url,
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": self.settings.llm_temperature,
                    "top_p": self.settings.llm_top_p,
                    "max_tokens": self.settings.llm_max_new_tokens,
                    "presence_penalty": self.settings.llm_presence_penalty,
                    "frequency_penalty": self.settings.llm_frequency_penalty,
                    "stream": True,
                    "echo" : False
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

                        delta = chunk["choices"][0]["delta"]
                        token = delta.get("content", "")

                        if token:
                            print(token, end="", flush=True)
                            full_response += token

                    except Exception as e:
                        print(f"\n[DEBUG] Failed to parse chunk: {data}")
                        print(f"[DEBUG] Error: {e}")
                        continue

        except requests.exceptions.RequestException as e:
            print(f"\n[Error] Failed to connect to LLM server: {e}")
            return

        print()

        self.chat_history.append(("user", user_text_raw))
        if full_response.strip():
            self.chat_history.append(("assistant", full_response.strip()))
        
        
        
    def _handle_chatbot_exit(self):
        print("Exiting chatbot...")


    def run(self):
        self.running = True

        print("\n===============================\n")
        print("Law Investment Chatbot")
        print("Type /help for commands.\n")
        
        print("\nAsistent: Kako vam lahko pomagam?")

        while self.running:
            try:
                user_input_string = input("\n>> ").strip()

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