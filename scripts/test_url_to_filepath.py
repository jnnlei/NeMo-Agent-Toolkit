#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify URL to filepath conversion fixes the directory issue.
"""

from web_utils import get_file_path_from_url


def test_url_to_filepath():
    """Test various URL patterns to ensure proper file path generation."""
    
    test_cases = [
        # (URL, expected_filename_part)
        ("https://docs.nvidia.com/", "index.html"),
        ("https://docs.nvidia.com", "index.html"),
        ("https://docs.nvidia.com/nim/introduction.html", "nim_introduction.html"),
        ("https://docs.nvidia.com/cuda/cuda-c-programming-guide/", "cuda_cuda-c-programming-guide_"),
        ("https://example.com/path/to/page.html", "path_to_page.html"),
    ]
    
    print("=" * 70)
    print("Testing URL to Filepath Conversion")
    print("=" * 70)
    
    all_passed = True
    for url, expected_part in test_cases:
        filepath, directory = get_file_path_from_url(url, "./.tmp/data")
        filename = filepath.split("/")[-1]
        
        # Check if it's a proper file (not ending with /)
        is_file = not filepath.endswith("/")
        
        status = "✓ PASS" if is_file else "✗ FAIL"
        if not is_file:
            all_passed = False
        
        print(f"\n{status}")
        print(f"  URL:       {url}")
        print(f"  Filepath:  {filepath}")
        print(f"  Filename:  {filename}")
        print(f"  Is File:   {is_file}")
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ All tests PASSED - No directory paths generated!")
    else:
        print("✗ Some tests FAILED - Directory paths detected!")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = test_url_to_filepath()
    exit(0 if success else 1)

