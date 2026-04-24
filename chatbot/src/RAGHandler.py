


class RAGHandler():
    
    def __init__(self, settings):
        
        self.settings = settings

    
    def build_RAG_prompt(self, user_text, chat_history):
        prompt = ""
        for role, content in chat_history:
            prompt += f'{role.capitalize()}: {content}'
        prompt += f"User: {user_text}\nAssistant: "
        return prompt