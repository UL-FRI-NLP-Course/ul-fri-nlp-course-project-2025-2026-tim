SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"


if [[ -f "${ENV_FILE}" ]]; then
    set -a
    source "${ENV_FILE}"
    set +a
fi

LLM_MODELS_DIR="${LLM_MODELS_DIR:-/d/hpc/projects/onj_fri/group-tim}"

DOWNLOAD_LINK='https://drive.usercontent.google.com/download?id=1L7q5QljWioW36koHG9U3wlDNF1YhKWQG&export=download&confirm=t'

mkdir -p "$LLM_MODELS_DIR/data/rag_store"

curl -L "$DOWNLOAD_LINK" \
-o "$LLM_MODELS_DIR/data/rag_store/download.zip"

unzip "$LLM_MODELS_DIR/data/rag_store/download.zip" -d "$LLM_MODELS_DIR/data/rag_store"
rm "$LLM_MODELS_DIR/data/rag_store/download.zip"