#!/usr/bin/env python3
"""
Build indices for the Contract RAG system.

Parses documents, chunks them, generates embeddings, and builds vector and BM25 indices.
"""

import argparse
from pathlib import Path

from src.parsers.document_parser import DocumentParser
from src.parsers.chunker import DocumentChunker
from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from config.settings import settings, setup_directories


def build_index(
    documents_dir: str = None,
    rebuild: bool = False,
    recursive: bool = False
):
    """
    Build all indices from documents.

    Args:
        documents_dir: Directory containing documents
        rebuild: Whether to rebuild from scratch (delete existing collection)
        recursive: Whether to search for documents recursively
    """
    print("=" * 70)
    print("Contract RAG - Building Index")
    print("=" * 70)

    # Setup directories
    setup_directories()

    documents_dir = documents_dir or settings.document.documents_dir

    # Initialize components
    print("\n[1/6] Initializing components...")
    parser = DocumentParser(documents_dir=documents_dir)
    chunker = DocumentChunker()
    embedder = Embedder()
    vector_store = VectorStore(embedding_dim=embedder.get_embedding_dimension())

    # Check if rebuild is needed
    if rebuild and vector_store.collection_exists():
        print("\n[!] Rebuilding index - deleting existing collection...")
        vector_store.delete_collection()
        vector_store._ensure_collection()

    # Parse documents
    print(f"\n[2/6] Parsing documents from {documents_dir}...")
    print(f"Supported formats: {', '.join(settings.document.supported_formats)}")

    documents = parser.parse_directory(
        directory=documents_dir,
        recursive=recursive
    )

    if not documents:
        print("\nNo documents found!")
        print(f"Please add documents to: {documents_dir}")
        print(f"Supported formats: {', '.join(settings.document.supported_formats)}")
        return

    print(f"\nParsed {len(documents)} document(s)")

    # Chunk documents
    print(f"\n[3/6] Chunking documents...")
    print(f"Chunk size: {chunker.chunk_size}, Overlap: {chunker.chunk_overlap}")

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

    # Show chunk distribution
    doc_chunk_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("source_file", "Unknown")
        doc_chunk_counts[source] = doc_chunk_counts.get(source, 0) + 1

    print("\nChunk distribution:")
    for doc_name, count in doc_chunk_counts.items():
        print(f"  {doc_name}: {count} chunks")

    # Generate embeddings
    print(f"\n[4/6] Generating embeddings...")
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_batch(chunk_texts, show_progress=True)

    print(f"Generated {len(embeddings)} embeddings")

    # Build vector store
    print(f"\n[5/6] Building vector store...")

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

    # Build BM25 index
    print(f"\n[6/6] Building BM25 index...")
    hybrid_search = HybridSearch(embedder=embedder, vector_store=vector_store)

    bm25_docs = [
        {"id": chunk.id, "text": chunk.text}
        for chunk in chunks
    ]

    hybrid_search.build_bm25_index(bm25_docs, save=True)

    # Summary
    print("\n" + "=" * 70)
    print("Index Build Complete!")
    print("=" * 70)
    print(f"\nTotal documents: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Vector store: {vector_store.get_document_count()} vectors")
    print(f"BM25 index: {len(hybrid_search.bm25_corpus)} documents")

    print(f"\nVector DB: {settings.qdrant.host}:{settings.qdrant.port}")
    print(f"Collection: {settings.qdrant.collection_name}")
    print(f"BM25 index: {hybrid_search.bm25_index_path}")

    print("\nYou can now run the chatbot with: python main.py")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Build indices for Contract RAG system"
    )

    parser.add_argument(
        "--documents-dir",
        type=str,
        default=None,
        help=f"Directory containing documents (default: {settings.document.documents_dir})"
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild from scratch (delete existing collection)"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for documents recursively in subdirectories"
    )

    args = parser.parse_args()

    try:
        build_index(
            documents_dir=args.documents_dir,
            rebuild=args.rebuild,
            recursive=args.recursive
        )
    except KeyboardInterrupt:
        print("\n\nBuild interrupted by user")
    except Exception as e:
        print(f"\n\nError during index build: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
