#!/usr/bin/env python3
"""
Interactive chatbot for querying contract documents.
"""

import argparse
import sys

from src.chatbot.contract_chatbot import ContractChatbot
from config.settings import settings


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("Contract RAG Chatbot")
    print("=" * 70)
    print("\nAsk questions about your contract documents.")
    print("Type 'exit', 'quit', or 'q' to exit.")
    print("Type 'help' for available commands.")
    print("=" * 70 + "\n")


def print_help():
    """Print help message."""
    print("\nAvailable commands:")
    print("  help, h, ?     - Show this help message")
    print("  exit, quit, q  - Exit the chatbot")
    print("  info           - Show system information")
    print("  clear          - Clear query cache")
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


def interactive_mode(chatbot: ContractChatbot, show_sources: bool = True):
    """
    Run the chatbot in interactive mode.

    Args:
        chatbot: ContractChatbot instance
        show_sources: Whether to show source passages
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
                else:
                    print("\nQuery cache is disabled.")
                continue

            # Process question
            print()
            response = chatbot.ask(question, return_passages=show_sources)

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

            print("\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            print()


def single_query_mode(chatbot: ContractChatbot, question: str, show_sources: bool = True):
    """
    Answer a single question and exit.

    Args:
        chatbot: ContractChatbot instance
        question: Question to answer
        show_sources: Whether to show source passages
    """
    # Check if index exists
    vs_info = chatbot.vector_store.get_collection_info()
    if not vs_info.get('exists') or vs_info.get('points_count', 0) == 0:
        print("ERROR: No documents indexed!")
        print("Please run 'python build_index.py' first to index your documents.")
        sys.exit(1)

    # Process question
    response = chatbot.ask(question, return_passages=show_sources)

    # Print answer
    print(response["answer"])

    # Show sources if enabled
    if show_sources and response.get("passages"):
        print(f"\n\nSources ({response['num_passages']} passages):")
        for i, passage in enumerate(response["passages"], 1):
            source = passage.get("source_file", "Unknown")
            print(f"[{i}] {source}")


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chatbot for contract documents"
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

    args = parser.parse_args()

    try:
        # Initialize chatbot
        print("Initializing chatbot...")
        chatbot = ContractChatbot(
            use_reranking=not args.no_rerank,
            use_cache=not args.no_cache
        )

        # Run in appropriate mode
        if args.question:
            single_query_mode(
                chatbot,
                args.question,
                show_sources=not args.no_sources
            )
        else:
            interactive_mode(
                chatbot,
                show_sources=not args.no_sources
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
