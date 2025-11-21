#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to clean up Milvus collections - delete or drop collections.
"""

import argparse
import sys

from pymilvus import connections, utility, Collection


def list_collections(uri: str):
    """List all collections in Milvus."""
    host, port = parse_uri(uri)
    connections.connect("default", host=host, port=port)
    
    collections = utility.list_collections()
    print("\n" + "=" * 70)
    print("Available Collections in Milvus")
    print("=" * 70)
    
    if not collections:
        print("No collections found.")
        return
    
    for coll_name in collections:
        try:
            collection = Collection(coll_name)
            # Flush to ensure all data is persisted
            collection.flush()
            collection.load()
            num_entities = collection.num_entities
            print(f"\n  📦 {coll_name}")
            print(f"     Documents: {num_entities}")
        except Exception as e:
            print(f"\n  📦 {coll_name}")
            print(f"     Error: {e}")
    
    print("\n" + "=" * 70)
    connections.disconnect("default")


def drop_collection(uri: str, collection_name: str, confirm: bool = False):
    """Drop (delete) a collection completely."""
    host, port = parse_uri(uri)
    connections.connect("default", host=host, port=port)
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist.")
        connections.disconnect("default")
        return False
    
    # Get collection info before dropping
    collection = Collection(collection_name)
    try:
        collection.load()
        num_entities = collection.num_entities
    except Exception:
        num_entities = "unknown"
    
    print("\n" + "=" * 70)
    print(f"⚠️  WARNING: About to DROP collection '{collection_name}'")
    print("=" * 70)
    print(f"  Collection: {collection_name}")
    print(f"  Documents: {num_entities}")
    print(f"  This action is IRREVERSIBLE!")
    print("=" * 70)
    
    if not confirm:
        response = input("\nType 'yes' to confirm deletion: ").strip().lower()
        if response != 'yes':
            print("❌ Cancelled.")
            connections.disconnect("default")
            return False
    
    try:
        utility.drop_collection(collection_name)
        print(f"\n✅ Successfully dropped collection '{collection_name}'")
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"\n❌ Error dropping collection: {e}")
        connections.disconnect("default")
        return False


def clear_collection(uri: str, collection_name: str, confirm: bool = False):
    """Clear all data from a collection (but keep the collection structure)."""
    host, port = parse_uri(uri)
    connections.connect("default", host=host, port=port)
    
    if not utility.has_collection(collection_name):
        print(f"❌ Collection '{collection_name}' does not exist.")
        connections.disconnect("default")
        return False
    
    collection = Collection(collection_name)
    collection.load()
    num_entities = collection.num_entities
    
    print("\n" + "=" * 70)
    print(f"⚠️  WARNING: About to CLEAR all data from '{collection_name}'")
    print("=" * 70)
    print(f"  Collection: {collection_name}")
    print(f"  Documents to delete: {num_entities}")
    print(f"  Collection structure will be preserved")
    print("=" * 70)
    
    if not confirm:
        response = input("\nType 'yes' to confirm clearing: ").strip().lower()
        if response != 'yes':
            print("❌ Cancelled.")
            connections.disconnect("default")
            return False
    
    try:
        # Delete all entities using expression that matches all
        collection.delete(expr="pk >= 0")
        collection.flush()
        print(f"\n✅ Successfully cleared all data from collection '{collection_name}'")
        print(f"   Deleted {num_entities} documents")
        connections.disconnect("default")
        return True
    except Exception as e:
        print(f"\n❌ Error clearing collection: {e}")
        print("\nTrying alternative method (drop and recreate)...")
        connections.disconnect("default")
        # If clear fails, suggest dropping
        return False


def parse_uri(uri: str):
    """Parse Milvus URI into host and port."""
    if "://" in uri:
        uri = uri.split("://")[1]
    
    if ":" in uri:
        host, port = uri.split(":")
        port = int(port)
    else:
        host = uri
        port = 19530
    
    return host, port


def main():
    parser = argparse.ArgumentParser(
        description="Clean up Milvus collections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all collections
  python clean_milvus_collection.py --list

  # Drop a collection (delete completely)
  python clean_milvus_collection.py --drop nim_docs

  # Clear all data but keep collection structure
  python clean_milvus_collection.py --clear nim_docs

  # Use with custom Milvus URI
  python clean_milvus_collection.py --list --uri http://192.168.1.100:19530
        """
    )
    
    parser.add_argument("--uri", "-u",
                       default="http://localhost:19530",
                       help="Milvus URI (default: http://localhost:19530)")
    
    parser.add_argument("--list", "-l",
                       action="store_true",
                       help="List all collections")
    
    parser.add_argument("--drop",
                       metavar="COLLECTION",
                       help="Drop (delete) a collection completely")
    
    parser.add_argument("--clear",
                       metavar="COLLECTION",
                       help="Clear all data from a collection")
    
    parser.add_argument("--yes", "-y",
                       action="store_true",
                       help="Skip confirmation prompts")
    
    args = parser.parse_args()
    
    # Must specify at least one action
    if not (args.list or args.drop or args.clear):
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.list:
            list_collections(args.uri)
        
        if args.drop:
            drop_collection(args.uri, args.drop, args.yes)
        
        if args.clear:
            success = clear_collection(args.uri, args.clear, args.yes)
            if not success:
                print("\n💡 Tip: If clearing fails, try dropping and recreating:")
                print(f"   python clean_milvus_collection.py --drop {args.clear}")
                print(f"   # Then re-run your ingestion script")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. Milvus is running")
        print("  2. URI is correct")
        print("  3. You have pymilvus installed: pip install pymilvus")
        sys.exit(1)


if __name__ == "__main__":
    main()

