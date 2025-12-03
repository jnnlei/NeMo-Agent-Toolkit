# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def generate_doc_id(article_id: str, chunk_index: int, content: str) -> str:
    """
    Generate a deterministic document ID based on article ID and chunk position.

    This ensures the same document chunk always gets the same ID, preventing duplicates
    when re-running the ingestion script.

    Args:
        article_id (str): Unique article ID from WixQA corpus.
        chunk_index (int): Index of the chunk in the document.
        content (str): Content of the chunk (used for additional uniqueness).

    Returns:
        str: Deterministic document ID.
    """
    # Create a unique string combining article ID, chunk index, and a hash of content
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    id_string = f"{article_id}:chunk_{chunk_index}:{content_hash}"
    # Generate a stable ID from the string
    return hashlib.sha256(id_string.encode('utf-8')).hexdigest()


async def main(
    corpus_file: str = "/localhome/local-jilei/WixQA/wix_kb_corpus/wix_kb_corpus.jsonl",
    milvus_uri: str = "http://localhost:19530",
    collection_name: str = "wix_collection",
    embedding_model: str = "nvidia/nv-embedqa-e5-v5",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
):
    """
    Ingest WixQA knowledge base corpus into Milvus.

    Args:
        corpus_file: Path to WixQA corpus JSONL file
        milvus_uri: Milvus server URI
        collection_name: Name of the Milvus collection to create/use
        embedding_model: NVIDIA embedding model to use
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
    """

    # Initialize embedder
    logger.info(f"Initializing embedder: {embedding_model}")
    embedder = NVIDIAEmbeddings(model=embedding_model, truncate="END")

    # Create Milvus vector store
    logger.info(f"Connecting to Milvus at {milvus_uri}")
    vector_store = Milvus(
        embedding_function=embedder,
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
    )

    # Check if collection exists
    collection_existed_before = vector_store.col is not None

    if collection_existed_before:
        logger.info(f"Using existing Milvus collection: {collection_name}")
        # Get collection info for logging
        try:
            num_entities = vector_store.client.query(
                collection_name=collection_name, filter="", output_fields=["count(*)"]
            )
            entity_count = num_entities[0]["count(*)"] if num_entities else "unknown number of"
            logger.info(f"Collection '{collection_name}' contains {entity_count} documents")
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")
    else:
        logger.info(f"Collection '{collection_name}' does not exist, will be created when documents are added")

    # Load corpus from JSONL file
    logger.info(f"Loading corpus from {corpus_file}")
    documents = []
    article_count = 0

    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            article_count += 1

            # Create a Document for each article
            # Use the 'contents' field as main content
            content = article.get('contents', '')

            # Create metadata
            metadata = {
                'article_id': article.get('id', ''),
                'url': article.get('url', ''),
                'title': article.get('title', ''),
                'article_type': article.get('article_type', 'article'),
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

    logger.info(f"Loaded {article_count} articles from corpus")

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    logger.info(f"Splitting documents into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    all_chunks = []
    all_ids = []

    for doc in documents:
        chunks = text_splitter.split_documents([doc])
        article_id = doc.metadata['article_id']

        # Generate deterministic IDs for chunks
        for idx, chunk in enumerate(chunks):
            chunk_id = generate_doc_id(article_id, idx, chunk.page_content)
            all_chunks.append(chunk)
            all_ids.append(chunk_id)

    logger.info(f"Created {len(all_chunks)} chunks from {article_count} articles")
    logger.info(f"Average chunks per article: {len(all_chunks) / article_count:.2f}")

    # Add documents to Milvus in batches to show progress and avoid timeouts
    batch_size = 1000
    total_chunks = len(all_chunks)
    logger.info(f"Adding {total_chunks} chunks to Milvus collection '{collection_name}' in batches of {batch_size}...")
    
    doc_ids = []
    for i in range(0, total_chunks, batch_size):
        batch_chunks = all_chunks[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        
        try:
            batch_doc_ids = await vector_store.aadd_documents(documents=batch_chunks, ids=batch_ids)
            doc_ids.extend(batch_doc_ids)
            
            progress = min(i + batch_size, total_chunks)
            percentage = (progress / total_chunks) * 100
            logger.info(f"Progress: {progress}/{total_chunks} ({percentage:.1f}%) chunks ingested")
        except Exception as e:
            logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
            # Optional: decide whether to continue or break
            # continue 

    logger.info(f"Successfully ingested {len(doc_ids)} chunks into Milvus")

    # Final status check
    if collection_existed_before:
        logger.info(f"Successfully added documents to existing collection '{collection_name}'")
    else:
        logger.info(f"Successfully created collection '{collection_name}' and added {len(doc_ids)} chunks")

    return doc_ids


if __name__ == "__main__":
    import argparse
    import asyncio

    DEFAULT_CORPUS = "/localhome/local-jilei/WixQA/wix_kb_corpus/wix_kb_corpus.jsonl"
    DEFAULT_COLLECTION = "wix_collection"
    DEFAULT_URI = "http://localhost:19530"

    parser = argparse.ArgumentParser(
        description="Ingest WixQA knowledge base corpus into Milvus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--corpus_file",
        default=DEFAULT_CORPUS,
        help="Path to WixQA corpus JSONL file",
    )
    parser.add_argument(
        "--collection_name",
        "-n",
        default=DEFAULT_COLLECTION,
        help="Milvus collection name",
    )
    parser.add_argument(
        "--milvus_uri",
        "-u",
        default=DEFAULT_URI,
        help="Milvus server URI",
    )
    parser.add_argument(
        "--embedding_model",
        default="nvidia/nv-embedqa-e5-v5",
        help="NVIDIA embedding model to use",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Chunk size for text splitting",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Chunk overlap for text splitting",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            corpus_file=args.corpus_file,
            milvus_uri=args.milvus_uri,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )

