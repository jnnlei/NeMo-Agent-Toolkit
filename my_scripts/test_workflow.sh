# enter the environment
cd /localhome/local-jilei/NeMo-Agent-Toolkit
source .venv/bin/activate

# install the dependencies
uv pip install -e '.[langchain]'
uv sync --all-groups --all-extras

# cd src/milvus_rag_with_memory
# uv pip install -e .

# Start the Milvus docker compose [Skip this step if you already have Milvus running]
docker compose -f examples/deploy/docker-compose.milvus.yml up -d
# then run NeMo-Agent-Toolkit/scripts/bootstrap_milvus.sh to bootstrap the data

# add the api keys
if [ -f .env ]; then
    source .env
    echo "Loaded keys from .env"
fi

# run the workflow
nat run --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml --input "What are the GPU configuration options for running Qwen2.5 72B Instruct?"

nat run --config_file examples/RAG/simple_rag/configs/wix_config.yml --input "How do I change my site's color palette in ADI?"

# start the server
nat serve --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml

# start the server in a tmux session
tmux new -s backend

cd /localhome/local-jilei/NeMo-Agent-Toolkit
source .venv/bin/activate

nat serve --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml \
 --host 0.0.0.0 --port 8001 --workers 4
# ctrl + b, then d to detach from the tmux session

# list the tmux sessions
tmux ls
# attach to the tmux session
tmux attach -t backend
# ctrl + c to kill the server, then type 'exit' to detach from the tmux session


# test the server
curl --request POST \
  --url http://localhost:8001/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "what is NVIDIA NIM?",
    "use_knowledge_base": true
}'

# open the nat-ui
cd external/nat-ui
npm ci
# changed proxy/server.js line 335 to use the host 0.0.0.0 to allow external access
PORT=8000 npm run dev
# open forwarded localhost:3000 in the browser

# start the ui in a tmux session
tmux new -s ui
cd /localhome/local-jilei/NeMo-Agent-Toolkit/external/nat-ui
PORT=8000 npm run dev
# ctrl + b, then d to detach from the tmux session

# attach to the tmux session
tmux attach -t ui
# ctrl + c to kill the ui, then type 'exit' to detach from the tmux session

# test the api and ui
curl http://10.78.17.253:8001/docs # api docs
curl http://10.78.17.253:8000 # ui

# evaluate the local workflow with timestamped output directory
tmux new -s wix_eval

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOCAL_OUTPUT_DIR="eval_output/local_run_${TIMESTAMP}"
mkdir -p "${LOCAL_OUTPUT_DIR}"
echo "Saving local eval results to: ${LOCAL_OUTPUT_DIR}"

nat eval --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml \
  --override eval.general.output.dir "${LOCAL_OUTPUT_DIR}"

nat eval --config_file examples/RAG/simple_rag/configs/wix_config.yml \
  --override eval.general.output.dir "${LOCAL_OUTPUT_DIR}"

tmux attach -t wix_eval

# evaluate the local workflow
nat eval --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml \

# evaluate the remote workflow
nat eval --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml --endpoint http://localhost:8000

# optimize the workflow
tmux new -s optimize
nat optimize --config_file examples/RAG/simple_rag/configs/wix_config.yml
tmux attach -t optimize