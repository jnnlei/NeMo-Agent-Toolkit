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
export NVIDIA_API_KEY=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
export MEM0_API_KEY=m0-w3WNdSPofqjsO7ghRK3dqEjzkbTvmrgTDqkyjpwn
export TAVILY_API_KEY=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

# run the workflow
nat run --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml --input "what is vllm?"

# start the server
nat serve --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml

# test the server
curl --request POST \
  --url http://localhost:8000/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "what is NVIDIA NIM?",
    "use_knowledge_base": true
}'

# open the nat-ui
cd external/nat-ui
npm ci
npm run dev
# open forwarded localhost:3000 in the browser


# evaluate the local workflow with timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOCAL_OUTPUT_DIR="eval_output/local_run_${TIMESTAMP}"
mkdir -p "${LOCAL_OUTPUT_DIR}"
echo "Saving local eval results to: ${LOCAL_OUTPUT_DIR}"
nat eval --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml \
  --override eval.general.output.dir "${LOCAL_OUTPUT_DIR}"

# evaluate the local workflow
nat eval --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml

# evaluate the remote workflow
nat eval --config_file examples/RAG/simple_rag/configs/milvus_memory_rag_tools_config.yml --endpoint http://localhost:8000

