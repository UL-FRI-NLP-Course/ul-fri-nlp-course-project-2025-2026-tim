# Natural language processing course: `Investment and Tax Law Chatbot`

A domain-specific LLM chatbot for answering questions about Slovenian tax and investing law.


# INSTALLATION
- place data in `chatbot/data`
- run `bash create_enviroment.sh` to create container and its overlay file and install packages in `requirements.txt`
- run `bash run_llm_server.sh` to queue up the LLM server job, the server bootup details are outputted into `configs/server_boot_config.yaml` which are used by the chatbot client (this can be a non interractive batch job)
- run `bash run_chatbot_client.sh` to queue up the client for communicating with the LLM server - this job is required to be interractive to use the chatbot terminal