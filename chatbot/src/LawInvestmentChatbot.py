from src.LLM import LLM

class LawInvestmentChatbot:
    def __init__(self, settings):
        self.running = False
        self.settings = settings
        self.chat_history = []

        

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


    def _initialize_conversation(self):
        system_prompt = (
            "You are a knowledgeable assistant specialized in law and investment topics. "
            "Provide clear, accurate, and concise answers. "
            "If unsure, say you are not certain instead of guessing. "
            "Use professional language and explain concepts when necessary."
        )
        self.chat_history.append(("System", system_prompt))


    def _build_prompt(self):
        MAX_TURNS = 5
        
        prompt = ""
        recent_history = self.chat_history[-2 * MAX_TURNS:]

        for role, message in recent_history:
            prompt += f"{role}: {message}\n"

        prompt += "Assistant:"
        return prompt
    
    
    def _handle_question(self, text: str):
        self.chat_history.append(("User", text))

        prompt = self._build_prompt()
        response = self.LLM.generate(prompt)
        response = response.split("Assistant:")[-1].strip()
        self.chat_history.append(("Assistant", response))

        print("Assistant:", response)



    def _handle_chatbot_exit(self):
        print("Exiting chatbot...")

        
    def run(self):
        
        self.LLM = LLM(self.settings)
        self.LLM.download_if_missing()
        self.LLM.load_model()
        self.running = True
        
        self._initialize_conversation()
        
        print("\n===============================\n")
        print("Law Investment Chatbot")
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