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

import asyncio
import logging
import os
from typing import List, Set, Tuple
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


async def _wrap_request(f, url):
    try:
        resp = {"url": url, "content": (await f).text}
    except Exception as e:
        logger.exception("Error in _wrap_request for %s: %s", url, e, exc_info=True)
        resp = {"url": url, "content": None, "exception": f"{e}"}
    return resp


async def scrape(urls: list | str, headers: dict = None):
    """
    Retrieve the page content for a given list of urls.

    Args:
      urls (list): List of urls (or a single url string)
      headers (dict): Dictionary of headers to use in the request

    Returns (Tuple(list[dict], list[dict])): Tuple containing lists of dictionaries:
      "responses" which contains the urls and content of each successful request
      "failures" which contains the urls and exceptions for each unsuccessful request
    """
    headers = {'user-agent': 'Mozilla/5.0'} if not headers else headers
    urls = [urls] if isinstance(urls, str) else urls
    responses = []
    failures = []
    async with httpx.AsyncClient() as client:
        tasks = [_wrap_request(client.get(
            url,
            headers=headers,
        ), url) for url in urls]
        for response_future in asyncio.as_completed(tasks):
            response = await response_future
            if response:
                responses.append(response)
            else:
                failures.append(response)
    logger.debug(responses)
    return responses, failures


def get_file_path_from_url(url: str, base_path: str) -> str:
    """
    Generate a filepath based on the url, using the domain as the parent directory.

    Resulting filepaths take the form {base_path}/{domain}/{page_name}
    Examples:
       http://mydomain.com/articles/generative_ai -> {base_path}/mydomain/articles_generative_ai
       http://mydomain.com/ -> {base_path}/mydomain/index.html

    Args:
     url (str): The url from which to generate a file name
     base_path (str): The base path to build the new path from

    Returns:
     filepath (str): File path based generated from the URL
     directory (str): Path to the parent directory
    """
    short_url, domain = _get_short_url(url)
    short_url = short_url.replace("/", "_").strip("_")
    
    # If short_url is empty (root path), use default filename
    if not short_url:
        short_url = "index.html"
    
    domain = domain.replace("/", "_")
    directory = os.path.join(base_path, domain)
    file_path = os.path.join(base_path, domain, short_url)
    return file_path, directory


def cache_html(input_dict: dict, base_path="."):
    """
    Save HTML data to disk.

    Args:
     input_dict (dict): Dictionary of HTML content containnig the url and content
     base_path (str): Base path under which all directories and files will be created

    Returns
     input_dict (dict): Original input
     file_path (str): Path to the saved data
    """
    url = input_dict.get("url")
    data = input_dict.get("content")
    if not url or not data:
        logger.exception("Invalid input for saving to cache for: %s", input)
        return input_dict, None
    file_path, directory = get_file_path_from_url(url, base_path)

    os.makedirs(directory, exist_ok=True)
    try:
        with open(file_path, 'w', encoding="utf-8") as f:
            f.write(data)
    except Exception as e:
        logger.exception("Unable to save data for %s", url, exc_info=True)
        raise e
    return input_dict, file_path


def _get_short_url(url: str):
    path = url.split("://")[-1].split("www.")[-1]
    path_components = path.split("/")
    domain = path_components[0]
    short_url = "/".join(path_components[1:])
    return short_url, domain


def extract_links_from_html(html_content: str, base_url: str, allowed_domains: Set[str] = None) -> List[str]:
    """
    Extract all links from HTML content.

    Args:
        html_content (str): HTML content to parse.
        base_url (str): Base URL for resolving relative links.
        allowed_domains (Set[str]): Set of allowed domains. None means only same domain as base_url.

    Returns:
        List[str]: List of normalized absolute URLs.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    links = []
    base_domain = urlparse(base_url).netloc

    # If no allowed domains specified, default to same domain only
    if allowed_domains is None:
        allowed_domains = {base_domain}

    for tag in soup.find_all(['a', 'link']):
        href = tag.get('href')
        if not href:
            continue

        # Skip anchors, JavaScript, mailto, etc.
        if href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
            continue

        # Convert to absolute URL
        absolute_url = urljoin(base_url, href)
        parsed = urlparse(absolute_url)

        # Only keep http/https links
        if parsed.scheme not in ('http', 'https'):
            continue

        # Domain filtering
        if parsed.netloc not in allowed_domains:
            continue

        # Remove fragment (# part) and normalize
        clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean_url += f"?{parsed.query}"

        links.append(clean_url)

    return list(set(links))  # Deduplicate


async def scrape_recursive(start_urls: List[str],
                           max_depth: int = 2,
                           allowed_domains: Set[str] = None,
                           max_pages: int = 100) -> Tuple[List[dict], Set[str]]:
    """
    Recursively scrape web pages and their links.

    Args:
        start_urls (List[str]): List of starting URLs.
        max_depth (int): Maximum crawl depth (0 = start URLs only).
        allowed_domains (Set[str]): Set of allowed domains. None means domains from start_urls.
        max_pages (int): Maximum number of pages to crawl.

    Returns:
        Tuple[List[dict], Set[str]]: (List of successful responses, Set of all visited URLs).
    """
    # Automatically extract domains from start URLs
    if allowed_domains is None:
        allowed_domains = {urlparse(url).netloc for url in start_urls}

    visited: Set[str] = set()
    to_visit: List[Tuple[str, int]] = [(url, 0) for url in start_urls]  # (url, depth)
    all_responses = []

    logger.info("Starting recursive crawl, max_depth: %s, allowed_domains: %s", max_depth, allowed_domains)

    while to_visit and len(visited) < max_pages:
        # Get current batch (same depth)
        current_depth = to_visit[0][1]
        current_batch = []

        while to_visit and to_visit[0][1] == current_depth:
            url, depth = to_visit.pop(0)
            if url not in visited:
                current_batch.append(url)
                visited.add(url)

        if not current_batch:
            continue

        logger.info("Depth %s: crawling %s pages", current_depth, len(current_batch))

        # Batch scrape all pages at current depth
        responses, failures = await scrape(current_batch)
        all_responses.extend(responses)

        if failures:
            logger.warning("Failed to scrape %s URLs at depth %s", len(failures), current_depth)

        # If not at max depth, extract new links
        if current_depth < max_depth:
            for response in responses:
                if response.get('content'):
                    try:
                        new_links = extract_links_from_html(response['content'], response['url'], allowed_domains)

                        # Add unvisited links to queue
                        for link in new_links:
                            if link not in visited and len(visited) < max_pages:
                                to_visit.append((link, current_depth + 1))

                        logger.info("Extracted %s new links from %s", len(new_links), response['url'])
                    except Exception as e:
                        logger.warning("Failed to extract links from %s: %s", response['url'], e)

    logger.info("Recursive crawl completed, visited %s pages", len(visited))
    return all_responses, visited
