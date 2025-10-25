#!/usr/bin/env python3
"""
Enhanced interactive chatbot with query enhancement and document filtering.
"""

import argparse
import sys
from typing import Optional

from src.chatbot.contract_chatbot import ContractChatbot
from src.retrieval.query_enhancer import QueryEnhancer
from src.retrieval.document_filter import DocumentFilter
from config.settings import settings


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("Contract RAG Chatbot - Enhanced")
    print("=" * 70)
    print("\nAsk questions about your contract documents.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("Type 'help' for available commands.")
    print("=" * 70 + "\n")


def print_help():
    """Print help message."""
    print("\nAvailable commands:")
    print("  help, h, ?       - Show this help message")
    print("  exit, quit, q    - Exit the chatbot")
    print("  info             - Show system information")
    print("  clear            - Clear query cache")
    print("  docs             - List all documents")
    print("  filter <file>    - Filter results to specific document")
    print("  unfilter         - Remove document filter")
    print("  compare          - Compare two documents")
    print("  enhance on/off   - Toggle query enhancement")
    print("\nJust type your question to get an answer.")
    print()


def print_info(chatbot: ContractChatbot):
    """Print system information."""
    print("\n" + "=" * 70)
    print("System Information")
    print("=" * 70)

    # Vector store info
    vs_info = chatbot.vector_store.get_collection_info()
    print(f"\nVector Store:")
    print(f"  Collection: {settings.qdrant.collection_name}")
    print(f"  Status: {vs_info.get('status', 'Unknown')}")
    print(f"  Documents: {vs_info.get('points_count', 0)} chunks")

    # Model info
    print(f"\nModels:")
    print(f"  Embedding: {settings.embedding.model_name}")
    print(f"  Reranking: {settings.rerank.model_name if chatbot.use_reranking else 'Disabled'}")
    print(f"  LLM: {settings.ollama.model}")

    # Cache info
    if chatbot.query_cache:
        cache_info = chatbot.query_cache.get_cache_info()
        print(f"\nQuery Cache:")
        print(f"  Enabled: {cache_info.get('enabled', False)}")
        print(f"  Size: {cache_info.get('size', 0)} entries")
        print(f"  TTL: {cache_info.get('ttl', 0)}s")

    # Retrieval settings
    print(f"\nRetrieval Settings:")
    print(f"  Top-k: {settings.retrieval.top_k}")
    print(f"  Rerank top-k: {settings.retrieval.rerank_top_k}")
    print(f"  Hybrid alpha: {settings.retrieval.hybrid_alpha}")
    print(f"  Use reranking: {chatbot.use_reranking}")

    print("=" * 70 + "\n")


def list_documents(doc_filter: DocumentFilter):
    """List all documents in the system."""
    print("\n" + "=" * 70)
    print("Available Documents")
    print("=" * 70)

    documents = doc_filter.list_documents()

    if not documents:
        print("\nNo documents found.")
        return

    print(f"\nFound {len(documents)} document(s):\n")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc['source_file']}")
        print(f"   Type: {doc['file_type'].upper()}")
        print(f"   Size: {doc['file_size']:,} bytes")
        print(f"   Chunks: {doc['total_chunks']}")
        print(f"   ID: {doc['document_id'][:16]}...")
        print()

    print("=" * 70 + "\n")
    return documents


def compare_documents_mode(
    chatbot: ContractChatbot,
    doc_filter: DocumentFilter,
    documents: list
):
    """Run document comparison mode."""
    print("\n" + "=" * 70)
    print("Document Comparison Mode")
    print("=" * 70)

    if len(documents) < 2:
        print("\nNeed at least 2 documents for comparison.")
        return

    # Show available documents
    print("\nAvailable documents:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc['source_file']}")

    # Select documents
    try:
        doc1_idx = int(input("\nSelect first document (number): ")) - 1
        doc2_idx = int(input("Select second document (number): ")) - 1

        if doc1_idx < 0 or doc1_idx >= len(documents):
            print("Invalid first document number.")
            return
        if doc2_idx < 0 or doc2_idx >= len(documents):
            print("Invalid second document number.")
            return

        doc1 = documents[doc1_idx]
        doc2 = documents[doc2_idx]

        print(f"\nComparing:")
        print(f"  Document 1: {doc1['source_file']}")
        print(f"  Document 2: {doc2['source_file']}")

        # Get comparison query
        query = input("\nEnter comparison query: ").strip()
        if not query:
            print("No query provided.")
            return

        # Create filters
        filter1 = {'source_file': doc1['source_file']}
        filter2 = {'source_file': doc2['source_file']}

        # Compare
        print("\nSearching both documents...")
        comparison = doc_filter.compare_documents(
            filter1, filter2, query, top_k=3
        )

        # Show results
        print("\n" + "=" * 70)
        print(f"Results for: {query}")
        print("=" * 70)

        print(f"\n--- {doc1['source_file']} ---")
        if comparison['document1']['results']:
            for i, result in enumerate(comparison['document1']['results'], 1):
                print(f"\n[{i}] Score: {result['score']:.4f} | Chunk {result.get('chunk_index', '?')}")
                print(f"    {result['text'][:200]}...")
        else:
            print("No matching passages found.")

        print(f"\n--- {doc2['source_file']} ---")
        if comparison['document2']['results']:
            for i, result in enumerate(comparison['document2']['results'], 1):
                print(f"\n[{i}] Score: {result['score']:.4f} | Chunk {result.get('chunk_index', '?')}")
                print(f"    {result['text'][:200]}...")
        else:
            print("No matching passages found.")

        print("\n" + "=" * 70 + "\n")

    except (ValueError, KeyboardInterrupt):
        print("\nComparison cancelled.")


def interactive_mode(
    chatbot: ContractChatbot,
    show_sources: bool = True,
    use_enhancement: bool = False
):
    """
    Run the chatbot in interactive mode with enhanced features.

    Args:
        chatbot: ContractChatbot instance
        show_sources: Whether to show source passages
        use_enhancement: Whether to use query enhancement
    """
    print_banner()

    # Check if index exists
    vs_info = chatbot.vector_store.get_collection_info()
    if not vs_info.get('exists') or vs_info.get('points_count', 0) == 0:
        print("ERROR: No documents indexed!")
        print("Please run 'python build_index.py' first to index your documents.")
        print()
        return

    print(f"Loaded {vs_info.get('points_count', 0)} document chunks.\n")

    # Initialize enhancement tools
    query_enhancer = QueryEnhancer() if use_enhancement else None
    doc_filter = DocumentFilter(chatbot.vector_store)
    documents = None
    current_filter = None

    while True:
        try:
            # Get user input
            question = input("Question: ").strip()

            if not question:
                continue

            # Handle commands
            question_lower = question.lower()

            if question_lower in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break

            if question_lower in ['help', 'h', '?']:
                print_help()
                continue

            if question_lower == 'info':
                print_info(chatbot)
                continue

            if question_lower == 'clear':
                if chatbot.query_cache:
                    chatbot.query_cache.clear()
                    print("\nQuery cache cleared.")
                else:
                    print("\nQuery cache is disabled.")
                continue

            if question_lower == 'docs':
                documents = list_documents(doc_filter)
                continue

            if question_lower.startswith('filter '):
                filename = question[7:].strip()
                current_filter = {'source_file': filename}
                print(f"\nFiltering to: {filename}")
                continue

            if question_lower == 'unfilter':
                current_filter = None
                print("\nFilter removed.")
                continue

            if question_lower == 'compare':
                if documents is None:
                    documents = doc_filter.list_documents()
                compare_documents_mode(chatbot, doc_filter, documents)
                continue

            if question_lower == 'enhance on':
                use_enhancement = True
                query_enhancer = QueryEnhancer()
                print("\nQuery enhancement enabled.")
                continue

            if question_lower == 'enhance off':
                use_enhancement = False
                query_enhancer = None
                print("\nQuery enhancement disabled.")
                continue

            # Process question with optional enhancement
            print()
            if use_enhancement and query_enhancer:
                print("Enhancing query...")
                enhanced = query_enhancer.generate_multi_queries(question, num_queries=3)
                print(f"Generated {len(enhanced)} query variations.")
                print()

                # Use all queries and merge results
                all_results = []
                for q in enhanced:
                    results = chatbot.retrieve_context(
                        q,
                        filter_conditions=current_filter
                    )
                    all_results.extend(results)

                # Deduplicate by ID and keep highest scores
                seen_ids = {}
                for result in all_results:
                    rid = result['id']
                    if rid not in seen_ids or result.get('rerank_score', 0) > seen_ids[rid].get('rerank_score', 0):
                        seen_ids[rid] = result

                passages = list(seen_ids.values())
                passages.sort(
                    key=lambda x: x.get('rerank_score', x.get('hybrid_score', 0)),
                    reverse=True
                )
                passages = passages[:settings.retrieval.rerank_top_k]

                # Generate response with original question
                response = chatbot.generate_response(question, passages)
            else:
                response = chatbot.ask(
                    question,
                    filter_conditions=current_filter,
                    return_passages=show_sources
                )

            # Print answer
            print("Answer:")
            print("-" * 70)
            print(response["answer"])
            print("-" * 70)

            # Show sources if enabled
            if show_sources and response.get("passages"):
                print(f"\nSources ({response['num_passages']} passages):")
                for i, passage in enumerate(response["passages"], 1):
                    source = passage.get("source_file", "Unknown")
                    chunk_idx = passage.get("chunk_index", "?")

                    score_parts = []
                    if "rerank_score" in passage:
                        score_parts.append(f"Rerank: {passage['rerank_score']:.3f}")
                    if "hybrid_score" in passage:
                        score_parts.append(f"Hybrid: {passage['hybrid_score']:.3f}")

                    score_str = " | ".join(score_parts) if score_parts else "N/A"

                    print(f"\n[{i}] {source} (Chunk {chunk_idx}) | {score_str}")
                    print(f"    {passage.get('text', '')[:200]}...")

            if current_filter:
                print(f"\n(Filtered to: {current_filter})")

            print("\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced chatbot with query enhancement and document filtering"
    )

    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Ask a single question and exit"
    )

    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source passages"
    )

    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable re-ranking"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable query caching"
    )

    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Enable query enhancement (multi-query)"
    )

    parser.add_argument(
        "--filter",
        type=str,
        help="Filter to specific document (filename)"
    )

    parser.add_argument(
        "--list-docs",
        action="store_true",
        help="List all documents and exit"
    )

    args = parser.parse_args()

    try:
        # Initialize chatbot
        print("Initializing chatbot...")
        chatbot = ContractChatbot(
            use_reranking=not args.no_rerank,
            use_cache=not args.no_cache
        )

        # List documents mode
        if args.list_docs:
            doc_filter = DocumentFilter(chatbot.vector_store)
            list_documents(doc_filter)
            return

        # Single question mode
        if args.question:
            filter_cond = None
            if args.filter:
                filter_cond = {'source_file': args.filter}

            response = chatbot.ask(
                args.question,
                filter_conditions=filter_cond,
                return_passages=not args.no_sources
            )

            print(response["answer"])

            if not args.no_sources and response.get("passages"):
                print(f"\n\nSources ({response['num_passages']} passages):")
                for i, passage in enumerate(response["passages"], 1):
                    source = passage.get("source_file", "Unknown")
                    print(f"[{i}] {source}")
        else:
            # Interactive mode
            interactive_mode(
                chatbot,
                show_sources=not args.no_sources,
                use_enhancement=args.enhance
            )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
