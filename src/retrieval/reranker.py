"""
Re-ranking module using cross-encoder models for improved relevance scoring.
"""

from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np

from config.settings import settings


class Reranker:
    """
    Cross-encoder based re-ranker for improving retrieval relevance.

    Cross-encoders jointly encode query and document pairs for
    more accurate relevance scoring than bi-encoder similarity.
    """

    def __init__(
        self,
        model_name: str = None,
        device: str = None
    ):
        """
        Initialize the re-ranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run model on (cpu/cuda/mps)
        """
        self.model_name = model_name or settings.rerank.model_name
        self.device = device or settings.rerank.device

        print(f"Loading re-ranking model: {self.model_name}")
        self.model = CrossEncoder(self.model_name, device=self.device)
        print(f"Re-ranking model loaded on {self.device}")

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = None,
        text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Re-rank search results using cross-encoder.

        Args:
            query: Search query
            results: List of search results with text
            top_k: Number of top results to return
            text_key: Key in result dict containing text to rank

        Returns:
            Re-ranked results with rerank scores
        """
        if not results:
            return []

        top_k = top_k or settings.retrieval.rerank_top_k

        # Extract texts from results
        texts = [result.get(text_key, "") for result in results]

        # Create query-document pairs
        pairs = [[query, text] for text in texts]

        # Compute cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Add rerank scores to results
        reranked_results = []
        for result, score in zip(results, scores):
            result_copy = result.copy()
            result_copy["rerank_score"] = float(score)
            reranked_results.append(result_copy)

        # Sort by rerank score
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return reranked_results[:top_k]

    def rerank_with_threshold(
        self,
        query: str,
        results: List[Dict[str, Any]],
        threshold: float = 0.0,
        top_k: int = None,
        text_key: str = "text"
    ) -> List[Dict[str, Any]]:
        """
        Re-rank and filter results by minimum score threshold.

        Args:
            query: Search query
            results: List of search results
            threshold: Minimum rerank score threshold
            top_k: Maximum number of results
            text_key: Key in result dict containing text

        Returns:
            Re-ranked and filtered results
        """
        reranked = self.rerank(query, results, top_k=len(results), text_key=text_key)

        # Filter by threshold
        filtered = [r for r in reranked if r["rerank_score"] >= threshold]

        # Apply top_k if specified
        if top_k is not None:
            filtered = filtered[:top_k]

        return filtered


def main():
    """Test the re-ranker."""
    print("=== Testing Re-ranker ===\n")

    # Initialize re-ranker
    reranker = Reranker()

    # Test query and documents
    query = "What are the payment terms in the contract?"

    results = [
        {
            "id": "chunk_001",
            "text": "Payment terms: The Company shall pay Service Provider within 30 days of invoice date.",
            "source_file": "contract_001.pdf",
            "initial_score": 0.85
        },
        {
            "id": "chunk_002",
            "text": "This Agreement is governed by the laws of the State of California.",
            "source_file": "contract_001.pdf",
            "initial_score": 0.45
        },
        {
            "id": "chunk_003",
            "text": "Late payments shall incur interest at 1.5% per month.",
            "source_file": "contract_001.pdf",
            "initial_score": 0.72
        },
        {
            "id": "chunk_004",
            "text": "Confidentiality: All proprietary information must remain confidential.",
            "source_file": "contract_002.pdf",
            "initial_score": 0.30
        },
        {
            "id": "chunk_005",
            "text": "Invoices must be submitted by the 5th of each month for payment processing.",
            "source_file": "contract_002.pdf",
            "initial_score": 0.68
        }
    ]

    print(f"Query: {query}\n")
    print(f"=== Before Re-ranking (sorted by initial_score) ===\n")
    for i, result in enumerate(sorted(results, key=lambda x: x["initial_score"], reverse=True), 1):
        print(f"{i}. Score: {result['initial_score']:.2f}")
        print(f"   Text: {result['text'][:80]}...")
        print()

    # Re-rank
    reranked = reranker.rerank(query, results, top_k=5)

    print(f"=== After Re-ranking (sorted by rerank_score) ===\n")
    for i, result in enumerate(reranked, 1):
        print(f"{i}. Rerank Score: {result['rerank_score']:.4f} (Initial: {result['initial_score']:.2f})")
        print(f"   Text: {result['text'][:80]}...")
        print()

    # Test with threshold
    print(f"=== Re-ranking with threshold (>= 0.5) ===\n")
    filtered = reranker.rerank_with_threshold(query, results, threshold=0.5, top_k=3)
    print(f"Filtered to {len(filtered)} results above threshold\n")
    for i, result in enumerate(filtered, 1):
        print(f"{i}. Rerank Score: {result['rerank_score']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
        print()


if __name__ == "__main__":
    main()
