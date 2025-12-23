# Agentic RAG Workflow

This project implements an Agentic RAG (Retrieval-Augmented Generation) workflow designed for reading internal technical documentation. Currently, the public WixQA benchmark is used for testing and optimization. It employs a ReAct agent, Milvus for vector storage, and NVIDIA NIM for LLM and Information Retrieval services.

## Configuration

The core configuration is located at:
`examples/RAG/simple_rag/configs/wix_config.yml`

### Key Components

*   **Agent Architecture**: ReAct Agent (`_type: react_agent`)
*   **LLM**: `qwen/qwen3-next-80b-a3b-instruct` (via NVIDIA NIM)
*   **Embeddings**: `nvidia/nv-embedqa-e5-v5` (via NVIDIA NIM)
*   **Reranker**: `nvidia/llama-3.2-nv-rerankqa-1b-v2` (via NVIDIA NIM)
*   **Vector Database**: Milvus
*   **Memory**: Mem0 (`saas_memory`)

### Tools Available to the Agent

1.  **wix_retriever_tool_with_rerank**: Retrieves information from the Wix Help Center knowledge base with re-ranking capability using `nvidia/llama-3.2-nv-rerankqa-1b-v2`.
2.  **vllm_retriever_tool**: Retrieves information about vLLM.
3.  **web_search_tool**: Performs internet searches using Tavily.
4.  **code_generation_tool**: Generates Python code.
5.  **Memory Tools**: `add_memory` and `get_memory` for managing long-term user preferences.

## Prerequisites

1.  **Environment Setup**:
    Ensure you are in the project root and activate the virtual environment:
    ```bash
    cd NeMo-Agent-Toolkit
    source .venv/bin/activate
    ```

2.  **Install Dependencies**:
    ```bash
    uv pip install -e '.[langchain]'
    uv sync --all-groups --all-extras
    ```

3.  **Docker Services**:
    Start the Milvus vector database:
    ```bash
    docker compose -f examples/deploy/docker-compose.milvus.yml up -d
    ```
    *Note: You may need to bootstrap data into Milvus if not already present (e.g., using `scripts/bootstrap_milvus.sh`).*

4.  **API Keys**:
    Ensure your environment variables are set (e.g., in a `.env` file):
    *   `NVIDIA_API_KEY`
    *   `TAVILY_API_KEY` (if using web search)

## Usage

### Running the Workflow (CLI)

To run the agent with a specific query:

```bash
nat run --config_file examples/RAG/simple_rag/configs/wix_config.yml --input "How do I change my site's color palette in ADI?"
```

### Serving the Workflow (API)

To start a FastAPI server for the workflow:

```bash
nat serve --config_file examples/RAG/simple_rag/configs/wix_config.yml --host 0.0.0.0 --port 8001 --workers 4
```

API Documentation will be available at: `http://localhost:8001/docs`

### Testing the API

You can test the running server with `curl`:

```bash
curl --request POST \
  --url http://localhost:8001/generate \
  --header 'Content-Type: application/json' \
  --data '{
    "input_message": "Can I start accepting payments on my site while my Wix Payments account is still under verification?",
    "use_knowledge_base": true
}'
```

### User Interface

To run the frontend UI:

1.  Navigate to the UI directory:
    ```bash
    cd external/nat-ui
    ```

2.  Install dependencies:
    ```bash
    npm ci
    ```

3.  Start the development server:
    ```bash
    PORT=8000 npm run dev
    ```

4.  Access the UI in your browser at `http://localhost:8000`.

## Evaluation & Optimization

### Evaluation
To evaluate the workflow using the defined metrics (Accuracy, Groundedness, Relevance, Trajectory Accuracy):

```bash
nat eval --config_file examples/RAG/simple_rag/configs/wix_config.yml
```

### Optimization
To run the prompt optimizer (using Genetic Algorithm):

```bash
nat optimize --config_file examples/RAG/simple_rag/configs/wix_config.yml
```

## Reference Scripts

For a complete list of commands and testing flows, refer to:
`NeMo-Agent-Toolkit/my_scripts/test_workflow.sh`

