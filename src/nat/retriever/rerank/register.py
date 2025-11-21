# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import List

from langchain_core.embeddings import Embeddings
from pydantic import Field
from pydantic import HttpUrl
from pymilvus import MilvusClient

from nat.builder.builder import Builder
from nat.builder.builder import LLMFrameworkEnum
from nat.builder.retriever import RetrieverProviderInfo
from nat.cli.register_workflow import register_retriever_client
from nat.cli.register_workflow import register_retriever_provider
from nat.data_models.retriever import RetrieverBaseConfig
from nat.retriever.interface import Retriever
from nat.retriever.models import Document
from nat.retriever.models import RetrieverOutput

logger = logging.getLogger(__name__)


class MilvusRerankRetriever(Retriever):
    """
    A Milvus Retriever that incorporates a Reranking step.
    
    Workflow:
    1. Retrieve top_k * expansion_factor candidate documents using vector search.
    2. Rerank the candidates using the provided reranker model.
    3. Return the top_k documents after reranking.
    """

    def __init__(
        self,
        milvus_client: MilvusClient,
        embedder: Embeddings,
        reranker_model,
        content_field: str = "text",
        expansion_factor: int = 3,
    ) -> None:
        self.client = milvus_client
        self.embedder = embedder
        self.reranker = reranker_model
        self.content_field = content_field
        self.expansion_factor = expansion_factor
        self._bound_params = {}

    def bind(self, **kwargs) -> None:
        """Bind default parameters for search."""
        self._bound_params = kwargs

    async def search(
        self,
        query: str,
        collection_name: str = None,
        top_k: int = 5,
        **kwargs
    ) -> RetrieverOutput:
        """
        Execute search with reranking.
        """
        # Merge bound parameters
        params = {**self._bound_params, **kwargs}
        collection_name = collection_name or params.get('collection_name')
        
        if not collection_name:
            raise ValueError("collection_name is required")

        # Stage 1: Vector Search (Retrieval)
        candidate_k = top_k * self.expansion_factor
        logger.info(
            f"RerankRetriever: Retrieving {candidate_k} candidates for query: {query[:50]}..."
        )
        
        query_vector = self.embedder.embed_query(query)
        
        search_results = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=candidate_k,
            output_fields=[self.content_field],
            search_params=params.get('search_params', {"metric_type": "L2"}),
        )
        
        # Convert to Document objects
        candidates = []
        for hit in search_results[0]:
            doc = Document(
                page_content=hit['entity'].get(self.content_field, ""),
                metadata={
                    'distance': hit['distance'],
                    'id': hit['id'],
                },
                document_id=str(hit['id'])
            )
            candidates.append(doc)
        
        if not candidates:
            return RetrieverOutput(results=[])
        
        # Stage 2: Reranking
        logger.info(f"RerankRetriever: Reranking {len(candidates)} candidates")
        reranked_docs = await self._rerank(query, candidates, top_k)
        
        return RetrieverOutput(results=reranked_docs)

    async def _rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        """
        Rerank documents using the configured reranker model.
        """
        try:
            # 添加详细日志
            logger.info(f"🔄 Starting reranking with model: {type(self.reranker).__name__}")
            logger.info(f"📊 Input: {len(documents)} documents, Output: top {top_k}")
            
            # 使用 NVIDIARerank 的 compress_documents 方法
            if hasattr(self.reranker, 'compress_documents'):
                logger.info("✅ Using NVIDIARerank compress_documents API")
                
                # compress_documents 返回排序后的文档列表
                reranked_docs = await self.reranker.acompress_documents(
                    documents=documents,
                    query=query
                )
                
                # 添加 rerank_score 到 metadata（从 relevance_score 字段）
                for doc in reranked_docs:
                    if 'relevance_score' in doc.metadata:
                        doc.metadata['rerank_score'] = doc.metadata['relevance_score']
                
                # 打印分数范围
                scores = [doc.metadata.get('rerank_score', 0) for doc in reranked_docs]
                if scores:
                    logger.info(f"🎯 Rerank scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
                
                # compress_documents 已经排序，直接取前 top_k
                logger.info(f"✅ Reranking completed, returning top {top_k} documents")
                return reranked_docs[:top_k]
            
            # Fallback: 如果不是 NVIDIARerank
            else:
                logger.warning(f"⚠️ Reranker doesn't have compress_documents method, returning original order")
                return documents[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Reranking failed: {e}. Returning results from vector search.", exc_info=True)
            return documents[:top_k]

    async def _llm_based_rerank(self, query: str, texts: List[str]) -> List[float]:
        """Fallback using LLM prompting to score relevance."""
        scores = []
        for text in texts:
            prompt = f"""Rate the relevance of this passage to the query on a scale of 0 to 10.
Query: {query}
Passage: {text[:500]}...

Output only a single number (e.g. 7.5)."""
            
            try:
                response = await self.reranker.ainvoke(prompt)
                # Handle potential string output processing
                score_str = str(response).strip()
                # Extract number if mixed with text
                import re
                match = re.search(r"[-+]?\d*\.\d+|\d+", score_str)
                score = float(match.group()) if match else 0.0
            except Exception:
                score = 0.0
            
            scores.append(score)
        
        return scores


class MilvusRerankRetrieverConfig(RetrieverBaseConfig, name="milvus_rerank_retriever"):
    """Configuration for Milvus Retriever with Reranking."""
    uri: HttpUrl = Field(description="The uri of Milvus service")
    collection_name: str = Field(description="The name of the milvus collection to search")
    embedding_model: str = Field(description="The name of the embedding model to use")
    reranker_model_name: str = Field(
        default="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        description="The NVIDIA reranker model name (e.g., nvidia/llama-3.2-nv-rerankqa-1b-v2)"
    )
    top_k: int = Field(default=5, description="The number of results to return after reranking")
    expansion_factor: int = Field(default=3, description="Factor to multiply top_k by for initial retrieval")
    content_field: str = Field(default="text", description="Name of the primary field to store/retrieve")
    search_params: dict = Field(default={"metric_type": "L2"}, description="Search parameters for Milvus")


@register_retriever_provider(config_type=MilvusRerankRetrieverConfig)
async def milvus_rerank_retriever_provider(retriever_config: MilvusRerankRetrieverConfig, builder: Builder):
    """Register the Milvus Rerank Retriever as a provider."""
    yield RetrieverProviderInfo(
        config=retriever_config,
        description="A Milvus retriever with reranking capabilities for improved search relevance"
    )


@register_retriever_client(config_type=MilvusRerankRetrieverConfig, wrapper_type=None)
async def milvus_rerank_retriever_client(config: MilvusRerankRetrieverConfig, builder: Builder):
    import os
    from langchain_nvidia_ai_endpoints import NVIDIARerank
    
    # 1. Get Embedder
    embedder = await builder.get_embedder(
        embedder_name=config.embedding_model,
        wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    
    # 2. Initialize NVIDIARerank (专门的 Reranker 客户端)
    api_key = os.getenv("NVIDIA_API_KEY", "")
    reranker = NVIDIARerank(
        model=config.reranker_model_name,
        api_key=api_key,
    )
    logger.info(f"Initialized NVIDIARerank with model: {config.reranker_model_name}")
    
    # 3. Initialize Milvus Client
    milvus_client = MilvusClient(uri=str(config.uri))
    
    # 4. Initialize Custom Retriever
    retriever = MilvusRerankRetriever(
        milvus_client=milvus_client,
        embedder=embedder,
        reranker_model=reranker,
        content_field=config.content_field,
        expansion_factor=config.expansion_factor,
    )
    
    # 5. Bind default parameters
    retriever.bind(
        collection_name=config.collection_name,
        top_k=config.top_k,
        search_params=config.search_params
    )
    
    yield retriever

