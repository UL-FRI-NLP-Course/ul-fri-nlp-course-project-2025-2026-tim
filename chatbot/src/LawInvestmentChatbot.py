import requests
from pathlib import Path
import yaml

class LawInvestmentChatbot:
    def __init__(self, settings):
        self.running = False
        self.settings = settings
        self.chat_history = []

        self.session_id = "default"

        config_path = Path(settings.chatbot_dir_path) / settings.server_boot_file_name
        if not config_path.exists():
            raise FileNotFoundError(
                f"Server boot config not found: {config_path}"
            )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        try:
            base_url = config["full_url"]
        except KeyError:
            raise ValueError("Invalid server config: missing 'url'")

        self.server_url = f"{base_url}/generate_stream"

        #print(f"[INFO] Using LLM server endpoint: {self.server_url}")



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




    def _handle_question(self, text: str):
        self.chat_history.append(("User", text))

        print("Assistant:", end=" ", flush=True)
        full_response = ""

        try:
            response = requests.post(
                self.server_url,
                json={
                    "session_id": self.session_id,
                    "text": text
                },
                stream=True,
                timeout=300
            )

            response.raise_for_status()

            for chunk in response.iter_lines(decode_unicode=True):
                if chunk:
                    print(chunk, end="", flush=True)
                    full_response += chunk

        except requests.exceptions.RequestException as e:
            print(f"\n[Error] Failed to connect to LLM server: {e}")
            return

        print()

        full_response = full_response.split("Assistant:")[-1].strip()
        self.chat_history.append(("Assistant", full_response))



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