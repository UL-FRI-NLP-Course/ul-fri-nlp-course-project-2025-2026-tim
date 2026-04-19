# Natural language processing course: `Investment and Tax Law Chatbot`

A domain-specific LLM chatbot for answering questions about Slovenian tax and investing law.


# INSTALLATION
- place data in `chatbot/data`
- run `bash create_enviroment.sh` to create container and its overlay file and install packages in `requirements.txt`
- run `bash run_chatbot.sh` to queue up interractive chatbot job, check for queue with `squeue -u $USER`