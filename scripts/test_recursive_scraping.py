#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Simple test script to verify the recursive scraping functionality.
"""

import asyncio
from web_utils import extract_links_from_html, scrape_recursive


def test_extract_links():
    """Test link extraction from HTML."""
    html_content = """
    <html>
        <head><title>Test Page</title></head>
        <body>
            <a href="/page1.html">Page 1</a>
            <a href="https://example.com/page2.html">Page 2</a>
            <a href="https://other.com/page3.html">Other Domain</a>
            <a href="#anchor">Anchor</a>
            <a href="javascript:void(0)">JavaScript</a>
            <a href="mailto:test@example.com">Email</a>
        </body>
    </html>
    """

    base_url = "https://example.com/index.html"
    allowed_domains = {"example.com"}

    links = extract_links_from_html(html_content, base_url, allowed_domains)

    print("✓ Link extraction test:")
    print(f"  Found {len(links)} links:")
    for link in sorted(links):
        print(f"    - {link}")

    # Should extract only example.com links (not other.com, anchors, javascript, mailto)
    expected_links = {
        "https://example.com/page1.html",
        "https://example.com/page2.html",
    }

    if set(links) == expected_links:
        print("  ✓ PASSED: Correct links extracted")
    else:
        print("  ✗ FAILED: Expected links don't match")
        print(f"    Expected: {expected_links}")
        print(f"    Got: {set(links)}")


async def test_recursive_scrape_dry_run():
    """Test recursive scraping with a small example (dry run, won't actually scrape)."""
    print("\n✓ Recursive scraping dry run test:")
    print("  Configuration:")
    print("    - Start URLs: ['https://example.com']")
    print("    - Max depth: 2")
    print("    - Max pages: 10")
    print("    - Allowed domains: {'example.com'}")
    print("  Note: This is a dry run demo. Actual scraping requires network access.")


def main():
    """Run tests."""
    print("=" * 60)
    print("Testing Recursive Web Scraping Functionality")
    print("=" * 60)

    # Test 1: Link extraction
    test_extract_links()

    # Test 2: Recursive scraping (dry run)
    asyncio.run(test_recursive_scrape_dry_run())

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
    print("\nUsage examples:")
    print("\n1. Basic recursive scraping (depth 2, max 100 pages):")
    print("   python langchain_web_ingest.py \\")
    print("     --urls https://docs.nvidia.com/cuda/ \\")
    print("     --recursive \\")
    print("     --collection_name cuda_docs_recursive")
    print("\n2. Recursive with custom depth and page limit:")
    print("   python langchain_web_ingest.py \\")
    print("     --urls https://example.com \\")
    print("     --recursive \\")
    print("     --max_depth 3 \\")
    print("     --max_pages 50")
    print("\n3. Multiple allowed domains:")
    print("   python langchain_web_ingest.py \\")
    print("     --urls https://docs.nvidia.com \\")
    print("     --recursive \\")
    print("     --allowed_domains docs.nvidia.com \\")
    print("     --allowed_domains developer.nvidia.com")


if __name__ == "__main__":
    main()

