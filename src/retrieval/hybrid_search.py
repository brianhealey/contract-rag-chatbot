"""
Hybrid search combining vector similarity and BM25 keyword search.
"""

from typing import List, Dict, Any, Optional
import pickle
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from config.settings import settings


class HybridSearch:
    """
    Combines vector search (semantic) with BM25 search (keyword)
    for improved retrieval accuracy.
    """

    def __init__(
        self,
        embedder: Embedder = None,
        vector_store: VectorStore = None,
        alpha: float = None
    ):
        """
        Initialize hybrid search.

        Args:
            embedder: Embedder instance
            vector_store: VectorStore instance
            alpha: Weight for hybrid scoring (0=BM25 only, 1=vector only)
        """
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore(
            embedding_dim=self.embedder.get_embedding_dimension()
        )
        self.alpha = alpha if alpha is not None else settings.retrieval.hybrid_alpha

        self.bm25 = None
        self.bm25_corpus = []  # Store documents for BM25
        self.bm25_doc_ids = []  # Store document IDs

        self.bm25_index_path = Path(settings.document.data_dir) / "bm25_index.pkl"

    def build_bm25_index(
        self,
        documents: List[Dict[str, Any]],
        save: bool = True
    ):
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents with 'id' and 'text' keys
            save: Whether to save index to disk
        """
        print("Building BM25 index...")

        # Tokenize documents (simple whitespace tokenization)
        self.bm25_corpus = []
        self.bm25_doc_ids = []

        for doc in documents:
            text = doc.get("text", "")
            doc_id = doc.get("id", "")

            # Simple tokenization: lowercase and split
            tokens = text.lower().split()
            self.bm25_corpus.append(tokens)
            self.bm25_doc_ids.append(doc_id)

        # Build BM25 index
        self.bm25 = BM25Okapi(self.bm25_corpus)

        print(f"BM25 index built with {len(self.bm25_corpus)} documents")

        # Save index
        if save:
            self.save_bm25_index()

    def save_bm25_index(self):
        """Save BM25 index to disk."""
        self.bm25_index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'corpus': self.bm25_corpus,
                'doc_ids': self.bm25_doc_ids
            }, f)

        print(f"BM25 index saved to {self.bm25_index_path}")

    def load_bm25_index(self) -> bool:
        """
        Load BM25 index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.bm25_index_path.exists():
            print(f"BM25 index not found at {self.bm25_index_path}")
            return False

        with open(self.bm25_index_path, 'rb') as f:
            data = pickle.load(f)

        self.bm25 = data['bm25']
        self.bm25_corpus = data['corpus']
        self.bm25_doc_ids = data['doc_ids']

        print(f"BM25 index loaded with {len(self.bm25_corpus)} documents")
        return True

    def _normalize_scores(self, scores: List[float]) -> np.ndarray:
        """
        Normalize scores to [0, 1] range.

        Args:
            scores: List of scores

        Returns:
            Normalized scores
        """
        scores = np.array(scores)
        if len(scores) == 0 or scores.max() == scores.min():
            return scores

        return (scores - scores.min()) / (scores.max() - scores.min())

    def bm25_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with scores
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_bm25_index() first.")

        # Tokenize query
        query_tokens = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append({
                    "id": self.bm25_doc_ids[idx],
                    "score": float(scores[idx])
                })

        return results

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_conditions: Optional metadata filters

        Returns:
            List of results with scores
        """
        # Embed query
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(
            query_vector=query_embedding,
            top_k=top_k,
            filter_conditions=filter_conditions
        )

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        alpha: float = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and BM25.

        Args:
            query: Search query
            top_k: Number of final results
            alpha: Weight for hybrid scoring (overrides default)
            filter_conditions: Optional metadata filters

        Returns:
            List of results with hybrid scores
        """
        top_k = top_k or settings.retrieval.top_k
        alpha = alpha if alpha is not None else self.alpha

        # Get more results from each method for better fusion
        retrieval_k = top_k * 2

        # Perform both searches
        vector_results = self.vector_search(
            query,
            top_k=retrieval_k,
            filter_conditions=filter_conditions
        )
        bm25_results = self.bm25_search(query, top_k=retrieval_k)

        # Normalize scores
        vector_scores = {r["id"]: r["score"] for r in vector_results}
        bm25_scores = {r["id"]: r["score"] for r in bm25_results}

        # Get all unique document IDs
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

        # Normalize score arrays
        if vector_scores:
            v_scores_array = self._normalize_scores(list(vector_scores.values()))
            vector_scores_norm = dict(zip(vector_scores.keys(), v_scores_array))
        else:
            vector_scores_norm = {}

        if bm25_scores:
            b_scores_array = self._normalize_scores(list(bm25_scores.values()))
            bm25_scores_norm = dict(zip(bm25_scores.keys(), b_scores_array))
        else:
            bm25_scores_norm = {}

        # Compute hybrid scores
        hybrid_results = []
        for doc_id in all_ids:
            v_score = vector_scores_norm.get(doc_id, 0.0)
            b_score = bm25_scores_norm.get(doc_id, 0.0)

            # Weighted combination
            hybrid_score = alpha * v_score + (1 - alpha) * b_score

            # Get full document info from vector results (has all metadata)
            doc_info = next((r for r in vector_results if r["id"] == doc_id), None)
            if doc_info is None:
                # Document found by BM25 but not in vector results
                # Fetch it directly from vector store
                try:
                    point = self.vector_store.client.retrieve(
                        collection_name=self.vector_store.collection_name,
                        ids=[doc_id]
                    )
                    if point and len(point) > 0:
                        doc_info = {
                            "id": point[0].id,
                            "score": 0.0,  # No vector score for this doc
                            **point[0].payload
                        }
                    else:
                        # Still not found, skip this document
                        continue
                except Exception as e:
                    # If retrieval fails, skip this document
                    print(f"Warning: Could not retrieve document {doc_id}: {e}")
                    continue

            hybrid_results.append({
                **doc_info,
                "hybrid_score": hybrid_score,
                "vector_score": v_score,
                "bm25_score": b_score
            })

        # Sort by hybrid score and return top-k
        hybrid_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return hybrid_results[:top_k]


def main():
    """Test hybrid search."""
    print("=== Testing Hybrid Search ===\n")

    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore(embedding_dim=embedder.get_embedding_dimension())
    hybrid_search = HybridSearch(embedder, vector_store)

    # Test documents
    documents = [
        {
            "id": "chunk_001",
            "text": "Payment terms: The Company shall pay Service Provider within 30 days of invoice date.",
            "source_file": "contract_001.pdf"
        },
        {
            "id": "chunk_002",
            "text": "This Agreement is governed by the laws of the State of California.",
            "source_file": "contract_001.pdf"
        },
        {
            "id": "chunk_003",
            "text": "Termination clause: Either party may terminate with 60 days written notice.",
            "source_file": "contract_002.pdf"
        },
        {
            "id": "chunk_004",
            "text": "Confidentiality: All proprietary information must remain confidential.",
            "source_file": "contract_002.pdf"
        }
    ]

    # Build indices
    print("Building indices...")

    # BM25 index
    hybrid_search.build_bm25_index(documents)

    # Vector index
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_batch(texts)
    ids = [doc["id"] for doc in documents]
    payloads = [{"text": doc["text"], "source_file": doc["source_file"]} for doc in documents]
    vector_store.add_documents(ids, embeddings, payloads)

    # Test search
    query = "What are the payment requirements?"
    print(f"\n=== Search Query: {query} ===\n")

    # Hybrid search
    results = hybrid_search.hybrid_search(query, top_k=3)

    print(f"Top {len(results)} results:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"   Vector: {result.get('vector_score', 0):.4f}, BM25: {result.get('bm25_score', 0):.4f}")
        print(f"   Text: {result['text']}")
        print(f"   Source: {result['source_file']}")
        print()


if __name__ == "__main__":
    main()
