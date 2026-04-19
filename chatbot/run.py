from src.diagnostics import run_import_diagnostics
from src.LawInvestmentChatbot import LawInvestmentChatbot

if __name__ == '__main__':
    print("NLP Course Run Chatbot Job Startup")
    run_import_diagnostics()

    chatbot = LawInvestmentChatbot()
    chatbot.run()
    
    print('Closing run_chatbot.sh script - Ending Job')
    