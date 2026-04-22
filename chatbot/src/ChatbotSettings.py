
from dataclasses import dataclass
import yaml

@dataclass
class ChatbotSettings:
    model_dir_path: str = '/models'
    chatbot_dir_path: str = '/workspace'
    server_boot_file_name: str = 'server_boot_config.yaml'
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k_chunks: int = 50
    reorder_top_n_chunks: int = 10
    nn_device: str = "cuda"
    
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_top_k_sampling: int = 50
    llm_repetition_penalty: float = 1.1
    llm_do_sample: bool = True
    llm_max_new_tokens: int = 256

    def __str__(self):
        return (
            "ChatbotSettings:\n"
            f"  model_dir_path: {self.model_dir_path}\n"
            f"  chatbot_dir_path: {self.chatbot_dir_path}\n"
            f"  server_boot_file_name: {self.server_boot_file_name}\n"
            f"  embedding_model: {self.embedding_model}\n"
            f"  LLM_model: {self.LLM_model}\n"
            f"  top_k_chunks: {self.top_k_chunks}\n"
            f"  reorder_top_n_chunks: {self.reorder_top_n_chunks}\n"
            f"  nn_device: {self.nn_device}\n"
            f"  llm_temperature: {self.llm_temperature}\n"
            f"  llm_top_p: {self.llm_top_p}\n"
            f"  llm_top_k_sampling: {self.llm_top_k_sampling}\n"
            f"  llm_repetition_penalty: {self.llm_repetition_penalty}\n"
            f"  llm_do_sample: {self.llm_do_sample}\n"
            f"  llm_max_new_tokens: {self.llm_max_new_tokens}"
        )

    
def load_settings(path: str) -> ChatbotSettings:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return ChatbotSettings(**data)
