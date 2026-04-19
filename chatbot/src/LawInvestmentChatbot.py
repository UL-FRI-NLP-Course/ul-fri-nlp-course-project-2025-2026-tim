

class LawInvestmentChatbot:
    def __init__(self):
        self.running = True

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
        print("Question answering not implemented!")

    def _handle_chatbot_exit(self):
        print("Exiting chatbot...")

        
    def run(self):
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