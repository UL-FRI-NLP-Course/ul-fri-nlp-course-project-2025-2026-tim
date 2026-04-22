from LawInvestmentChatbot import LawInvestmentChatbot
from ChatbotSettings import load_settings

CONFIG_FILE_PATH = "/workspace/config.yaml"

if __name__ == '__main__':
    print("NLP Course Run Chatbot Job Startup")
    
    print(f'Loading Settings from: {CONFIG_FILE_PATH}')
    settings = load_settings(CONFIG_FILE_PATH)
    print(settings)
    print("")
    
    print(f'Starting Chatbot!\n')

    chatbot = LawInvestmentChatbot(settings)
    chatbot.run()
    
    print('Closing run_chatbot.sh script - Ending Job')
    