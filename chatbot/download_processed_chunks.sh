
LLM_MODELS_DIR=/d/hpc/projects/onj_fri/group-tim

DOWNLOAD_LINK='https://drive.usercontent.google.com/download?id=1L7q5QljWioW36koHG9U3wlDNF1YhKWQG&export=download&confirm=t'

curl -L "$DOWNLOAD_LINK" \
-o "$LLM_MODELS_DIR/data/rag_store/download.zip"

unzip "$LLM_MODELS_DIR/data/rag_store/download.zip" -d "$LLM_MODELS_DIR"
rm "$LLM_MODELS_DIR/data/rag_store/download.zip"