# python scripts/langchain_web_ingest.py \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/introduction.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/getting-started.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/profiles.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/custom-decoding-backend.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/deploy-air-gap.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/deploy-helm.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/multi-node-deployment.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/deploy-behind-proxy.html \
#  --urls https://docs.nvidia.com/nim-operator/latest/index.html \
#  --urls https://docs.nvidia.com/nim-operator/latest/service.html \
#  --urls https://docs.nvidia.com/nim-operator/latest/cache.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/benchmarking.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/configuration.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/deterministic-mode.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/kv-cache-reuse.html \
#  --urls https://docs.nvidia.com/nim/large-language-models/latest/nim-repository-override.html \
#  --collection_name=nim_docs

# python scripts/langchain_web_ingest.py \
#  --urls https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/ \
#  --urls https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/ \
#  --collection_name=nim_docs

python scripts/langchain_web_ingest.py \
 --urls https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html \
 --collection_name=nim_docs