#!/bin/bash

# 清理有问题的缓存
rm -rf ./.tmp/data/docs.nvidia.com/
# Run with force to ignore cache and enable recursive crawling
python scripts/langchain_web_ingest.py \
  --urls https://docs.nvidia.com/nim/large-language-models/latest/introduction.html \
  --recursive \
  --force \
  --max_depth 3 \
  --max_pages 100 \
  --collection_name nim_docs

