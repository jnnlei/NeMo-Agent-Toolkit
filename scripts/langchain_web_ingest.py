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
import logging
import os
from uuid import uuid4

from langchain_community.document_loaders import BSHTMLLoader
from langchain_milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from web_utils import cache_html
from web_utils import get_file_path_from_url
from web_utils import scrape

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


def generate_doc_id(source_url: str, chunk_index: int, content: str) -> str:
    """
    Generate a deterministic document ID based on source URL and chunk position.

    This ensures the same document chunk always gets the same ID, preventing duplicates
    when re-running the ingestion script.

    Args:
        source_url (str): Source URL of the document.
        chunk_index (int): Index of the chunk in the document.
        content (str): Content of the chunk (used for additional uniqueness).

    Returns:
        str: Deterministic document ID.
    """
    # Create a unique string combining URL, chunk index, and a hash of content
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    id_string = f"{source_url}:chunk_{chunk_index}:{content_hash}"
    # Generate a stable ID from the string
    return hashlib.sha256(id_string.encode('utf-8')).hexdigest()


async def main(*,
               urls: list[str],
               milvus_uri: str,
               collection_name: str,
               clean_cache: bool = True,
               embedding_model: str = "nvidia/nv-embedqa-e5-v5",
               base_path: str = "./.tmp/data",
               recursive: bool = False,
               max_depth: int = 2,
               allowed_domains: list[str] = None,
               max_pages: int = 100,
               force: bool = False):

    embedder = NVIDIAEmbeddings(model=embedding_model, truncate="END")

    # Create the Milvus vector store
    vector_store = Milvus(
        embedding_function=embedder,
        collection_name=collection_name,
        connection_args={"uri": milvus_uri},
    )

    # Check if collection existed (Milvus connects to existing collections during init)
    collection_existed_before = vector_store.col is not None

    if collection_existed_before:
        logger.info("Using existing Milvus collection: %s", collection_name)
        # Get collection info for logging
        try:
            num_entities = vector_store.client.query(collection_name=collection_name,
                                                     filter="",
                                                     output_fields=["count(*)"])
            entity_count = num_entities[0]["count(*)"] if num_entities else "unknown number of"
            logger.info("Collection '%s' contains %s documents", collection_name, entity_count)
        except Exception as e:
            logger.warning("Could not get collection info: %s", e)
    else:
        logger.info("Collection '%s' does not exist, will be created when documents are added", collection_name)

    # Check which URLs are already cached
    if force:
        # Force mode: ignore cache and scrape all URLs
        logger.info("Force mode enabled: ignoring cache")
        cached_files = []
        urls_to_scrape = urls
    else:
        cached_files = [
            get_file_path_from_url(url, base_path)[0] for url in urls
            if os.path.exists(get_file_path_from_url(url, base_path)[0])
        ]
        urls_to_scrape = [url for url in urls if get_file_path_from_url(url, base_path)[0] not in cached_files]

        if cached_files:
            logger.info("Loading %s files from cache", len(cached_files))

    if recursive and urls_to_scrape:
        # Recursive crawling mode
        logger.info("Using recursive crawling mode (max_depth=%s, max_pages=%s)", max_depth, max_pages)
        from web_utils import scrape_recursive

        allowed_domains_set = set(allowed_domains) if allowed_domains else None
        html_data, visited_urls = await scrape_recursive(
            urls_to_scrape, max_depth=max_depth, allowed_domains=allowed_domains_set, max_pages=max_pages
        )
        logger.info("Recursive crawl completed: fetched %s pages, visited %s URLs", len(html_data), len(visited_urls))
        filenames = cached_files + [cache_html(data, base_path)[1] for data in html_data if data.get('content')]
    elif urls_to_scrape:
        # Standard non-recursive mode
        logger.info("Scraping %s URLs (non-recursive mode)", len(urls_to_scrape))
        html_data, err = await scrape(urls_to_scrape)
        if err:
            logger.warning("Failed to scrape %s URLs: %s", len(err), [f.get('url') for f in err])
        filenames = cached_files + [cache_html(data, base_path)[1] for data in html_data if html_data]
    else:
        # All URLs are cached
        filenames = cached_files

    doc_ids = []
    for filename in filenames:

        logger.info("Parsing %s into documents", filename)
        loader = BSHTMLLoader(filename)
        splitter = RecursiveCharacterTextSplitter()
        # splitter = MarkdownHeaderTextSplitter(
        #     headers_to_split_on=[
        #         ("#", "Header 1"),
        #         ("##", "Header 2"),
        #         ("###", "Header 3"),
        #     ]
        # )
        docs = loader.load()
        docs = splitter.split_documents(docs)

        if not isinstance(docs, list):
            docs = [docs]

        # Generate deterministic IDs based on source URL and chunk position
        # This prevents duplicate documents when re-running the script
        source_url = docs[0].metadata.get('source', filename) if docs else filename
        ids = [generate_doc_id(source_url, idx, doc.page_content) for idx, doc in enumerate(docs)]

        logger.info("Adding %s document chunks to Milvus collection %s", len(docs), collection_name)
        doc_ids.extend(await vector_store.aadd_documents(documents=docs, ids=ids))
        logger.info("Ingested %s document chunks", len(doc_ids))
        if clean_cache:
            logger.info("Removing %s", filename)
            os.remove(filename)

    # Final status check
    if collection_existed_before:
        logger.info("Successfully added %s new documents to existing collection '%s'", len(doc_ids), collection_name)
    else:
        logger.info("Successfully created collection '%s' and added %s new documents", collection_name, len(doc_ids))

    return doc_ids


if __name__ == "__main__":
    import argparse
    import asyncio

    CUDA_URLS = [
        "https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html",
        "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
        "https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html",
        "https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html",
    ]
    CUDA_COLLECTION_NAME = "cuda_docs"
    DEFAULT_URI = "http://localhost:19530"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--urls",
                        default=[],
                        action="append",
                        help="Urls to scrape for RAG context. Defaults to built-in URLs for NVIDIA CUDA documentation.")
    parser.add_argument("--collection_name", "-n", default=CUDA_COLLECTION_NAME, help="Collection name for the data.")
    parser.add_argument("--milvus_uri", "-u", default=DEFAULT_URI, help="Milvus host URI")
    parser.add_argument("--clean_cache", default=False, help="If true, deletes local files", action="store_true")
    parser.add_argument("--recursive",
                        "-r",
                        default=False,
                        action="store_true",
                        help="Enable recursive crawling mode to follow links on pages.")
    parser.add_argument("--max_depth",
                        "-d",
                        type=int,
                        default=2,
                        help="Maximum crawl depth for recursive mode (0 = start URLs only).")
    parser.add_argument("--allowed_domains",
                        default=[],
                        action="append",
                        help="Allowed domains for recursive crawling. Defaults to domains from start URLs.")
    parser.add_argument("--max_pages",
                        "-m",
                        type=int,
                        default=100,
                        help="Maximum number of pages to crawl in recursive mode.")
    parser.add_argument("--force",
                        "-f",
                        default=False,
                        action="store_true",
                        help="Force re-scraping even if URLs are already cached. Useful for recursive mode.")
    args = parser.parse_args()

    if len(args.urls) == 0:
        args.urls = CUDA_URLS

    asyncio.run(
        main(
            urls=args.urls,
            milvus_uri=args.milvus_uri,
            collection_name=args.collection_name,
            clean_cache=args.clean_cache,
            recursive=args.recursive,
            max_depth=args.max_depth,
            allowed_domains=args.allowed_domains if args.allowed_domains else None,
            max_pages=args.max_pages,
            force=args.force,
        ))
