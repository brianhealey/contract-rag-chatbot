#!/usr/bin/env python3
"""
Add new documents to existing indices incrementally.

This script allows you to add new documents without rebuilding the entire index.
"""

import argparse
from pathlib import Path
from typing import List

from src.parsers.document_parser import DocumentParser
from src.parsers.chunker import DocumentChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from config.settings import settings


def add_documents(file_paths: List[str]):
    """
    Add new documents to the index incrementally.

    Args:
        file_paths: List of file paths to add
    """
    print("=" * 70)
    print("Contract RAG - Adding Documents")
    print("=" * 70)

    # Validate files
    valid_files = []
    for file_path in file_paths:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}")
            continue

        file_type = path.suffix.lstrip('.').lower()
        if file_type not in settings.document.supported_formats:
            print(f"Warning: Unsupported format: {file_path} ({file_type})")
            continue

        valid_files.append(str(path))

    if not valid_files:
        print("\nNo valid files to add!")
        return

    print(f"\nAdding {len(valid_files)} document(s)...")

    # Initialize components
    print("\n[1/5] Initializing components...")
    parser = DocumentParser()
    chunker = DocumentChunker()
    embedder = Embedder()
    vector_store = VectorStore(embedding_dim=embedder.get_embedding_dimension())

    # Check if index exists
    if not vector_store.collection_exists():
        print("\nError: No existing index found!")
        print("Please run 'python build_index.py' first to create the initial index.")
        return

    initial_count = vector_store.get_document_count()
    print(f"Current index size: {initial_count} chunks")

    # Parse documents
    print(f"\n[2/5] Parsing documents...")
    documents = []
    for file_path in valid_files:
        try:
            doc = parser.parse_file(file_path)
            documents.append(doc)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            continue

    if not documents:
        print("\nNo documents successfully parsed!")
        return

    print(f"Parsed {len(documents)} document(s)")

    # Chunk documents
    print(f"\n[3/5] Chunking documents...")
    doc_dicts = [
        {
            "id": doc.id,
            "text": doc.text,
            "metadata": doc.metadata
        }
        for doc in documents
    ]

    chunks = chunker.chunk_documents(doc_dicts)
    print(f"Generated {len(chunks)} chunks")

    # Generate embeddings
    print(f"\n[4/5] Generating embeddings...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_batch(chunk_texts, show_progress=True)

    # Add to vector store
    print(f"\n[5/5] Adding to vector store...")
    chunk_ids = [chunk.id for chunk in chunks]
    chunk_payloads = [
        {
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "document_id": chunk.document_id,
            **chunk.metadata
        }
        for chunk in chunks
    ]

    vector_store.add_documents(chunk_ids, embeddings, chunk_payloads)

    # Update BM25 index
    print("\n[6/6] Updating BM25 index...")
    hybrid_search = HybridSearch(embedder=embedder, vector_store=vector_store)

    # Load existing BM25 index
    if hybrid_search.load_bm25_index():
        print("Loaded existing BM25 index")

        # Add new documents
        for chunk in chunks:
            tokens = chunk.text.lower().split()
            hybrid_search.bm25_corpus.append(tokens)
            hybrid_search.bm25_doc_ids.append(chunk.id)

        # Rebuild BM25 with updated corpus
        from rank_bm25 import BM25Okapi
        hybrid_search.bm25 = BM25Okapi(hybrid_search.bm25_corpus)

        # Save updated index
        hybrid_search.save_bm25_index()
    else:
        print("No existing BM25 index found. Creating new one...")
        bm25_docs = [{"id": chunk.id, "text": chunk.text} for chunk in chunks]
        hybrid_search.build_bm25_index(bm25_docs, save=True)

    # Summary
    final_count = vector_store.get_document_count()
    added_count = final_count - initial_count

    print("\n" + "=" * 70)
    print("Documents Added Successfully!")
    print("=" * 70)
    print(f"\nDocuments added: {len(documents)}")
    print(f"New chunks: {len(chunks)}")
    print(f"Previous index size: {initial_count} chunks")
    print(f"New index size: {final_count} chunks")
    print(f"Chunks added: {added_count}")

    print("\nAdded documents:")
    for doc in documents:
        print(f"  - {doc.source_file}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Add new documents to Contract RAG index incrementally"
    )

    parser.add_argument(
        "files",
        nargs="+",
        type=str,
        help="Paths to document files to add"
    )

    args = parser.parse_args()

    try:
        add_documents(args.files)
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user")
    except Exception as e:
        print(f"\n\nError adding documents: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
