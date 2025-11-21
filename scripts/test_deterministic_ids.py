#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify deterministic ID generation prevents duplicate documents.
"""

import sys
sys.path.insert(0, '.')

from langchain_web_ingest import generate_doc_id


def test_deterministic_ids():
    """Test that deterministic IDs are generated correctly."""
    
    print("=" * 70)
    print("Testing Deterministic Document ID Generation")
    print("=" * 70)
    
    # Test case 1: Same URL and content should generate same ID
    url1 = "https://docs.nvidia.com/cuda/guide.html"
    content1 = "This is the content of the first chunk."
    
    id1_run1 = generate_doc_id(url1, 0, content1)
    id1_run2 = generate_doc_id(url1, 0, content1)
    
    print("\n✓ Test 1: Same URL + content → Same ID")
    print(f"  URL:     {url1}")
    print(f"  Content: {content1[:50]}...")
    print(f"  ID Run 1: {id1_run1}")
    print(f"  ID Run 2: {id1_run2}")
    print(f"  Match: {id1_run1 == id1_run2} {'✓ PASS' if id1_run1 == id1_run2 else '✗ FAIL'}")
    
    # Test case 2: Different chunk index should generate different ID
    id2_chunk0 = generate_doc_id(url1, 0, content1)
    id2_chunk1 = generate_doc_id(url1, 1, content1)
    
    print("\n✓ Test 2: Same URL + content, different chunk → Different ID")
    print(f"  Chunk 0 ID: {id2_chunk0}")
    print(f"  Chunk 1 ID: {id2_chunk1}")
    print(f"  Different: {id2_chunk0 != id2_chunk1} {'✓ PASS' if id2_chunk0 != id2_chunk1 else '✗ FAIL'}")
    
    # Test case 3: Different URL should generate different ID
    url2 = "https://docs.nvidia.com/cuda/different.html"
    id3_url1 = generate_doc_id(url1, 0, content1)
    id3_url2 = generate_doc_id(url2, 0, content1)
    
    print("\n✓ Test 3: Different URL, same content → Different ID")
    print(f"  URL 1: {url1}")
    print(f"  URL 2: {url2}")
    print(f"  ID 1: {id3_url1}")
    print(f"  ID 2: {id3_url2}")
    print(f"  Different: {id3_url1 != id3_url2} {'✓ PASS' if id3_url1 != id3_url2 else '✗ FAIL'}")
    
    # Test case 4: Different content should generate different ID
    content2 = "This is different content."
    id4_content1 = generate_doc_id(url1, 0, content1)
    id4_content2 = generate_doc_id(url1, 0, content2)
    
    print("\n✓ Test 4: Same URL + chunk, different content → Different ID")
    print(f"  Content 1: {content1[:50]}...")
    print(f"  Content 2: {content2[:50]}...")
    print(f"  ID 1: {id4_content1}")
    print(f"  ID 2: {id4_content2}")
    print(f"  Different: {id4_content1 != id4_content2} {'✓ PASS' if id4_content1 != id4_content2 else '✗ FAIL'}")
    
    print("\n" + "=" * 70)
    print("Key Benefits:")
    print("  1. ✓ Same document → Same ID (prevents duplicates)")
    print("  2. ✓ Different documents → Different IDs (preserves uniqueness)")
    print("  3. ✓ Multiple runs → Same IDs (idempotent ingestion)")
    print("=" * 70)
    
    # Summary
    all_passed = (
        id1_run1 == id1_run2 and
        id2_chunk0 != id2_chunk1 and
        id3_url1 != id3_url2 and
        id4_content1 != id4_content2
    )
    
    print("\n" + ("✓ All tests PASSED!" if all_passed else "✗ Some tests FAILED!"))
    return all_passed


if __name__ == "__main__":
    success = test_deterministic_ids()
    exit(0 if success else 1)

