from src.diagnostics import run_import_diagnostics, run_file_system_diagnostics
from src.LawInvestmentChatbot import LawInvestmentChatbot
from src.ChatbotSettings import load_settings


CONFIG_FILE_PATH = "/workspace/config.yaml"
RUN_DIAGNOSTICS = True

if __name__ == '__main__':
    print("NLP Course Run Chatbot Job Startup")
    if RUN_DIAGNOSTICS:
        print("=== Running diagnostics ===")
        run_import_diagnostics()
        run_file_system_diagnostics()
        print("=== Diagnostics complete ===\n")
    
    print(f'Loading Settings from: {CONFIG_FILE_PATH}')
    settings = load_settings(CONFIG_FILE_PATH)
    print(settings)
    print("")
    
    print(f'Starting Chatbot!\n')

    chatbot = LawInvestmentChatbot(settings)
    chatbot.run()
    
    print('Closing run_chatbot.sh script - Ending Job')
    