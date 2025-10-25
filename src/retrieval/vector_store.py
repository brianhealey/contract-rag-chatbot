"""
Vector store implementation using Qdrant.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
)
from qdrant_client.http import models

from config.settings import settings


class VectorStore:
    """
    Wrapper for Qdrant vector database operations.
    """

    def __init__(
        self,
        collection_name: str = None,
        embedding_dim: int = None,
        host: str = None,
        port: int = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the Qdrant collection
            embedding_dim: Dimension of embedding vectors
            host: Qdrant host
            port: Qdrant port
        """
        self.collection_name = collection_name or settings.qdrant.collection_name
        self.embedding_dim = embedding_dim or settings.embedding.embedding_dimension
        self.host = host or settings.qdrant.host
        self.port = port or settings.qdrant.port

        # Initialize Qdrant client
        print(f"Connecting to Qdrant at {self.host}:{self.port}")
        self.client = QdrantClient(host=self.host, port=self.port)

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]

        if self.collection_name not in collection_names:
            print(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE
                ),
            )
            print(f"Collection '{self.collection_name}' created successfully.")
        else:
            print(f"Collection '{self.collection_name}' already exists.")

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        payloads: List[Dict[str, Any]],
    ) -> bool:
        """
        Add documents to the vector store.

        Args:
            ids: List of unique document IDs
            embeddings: List of embedding vectors
            payloads: List of metadata dictionaries

        Returns:
            True if successful
        """
        if not (len(ids) == len(embeddings) == len(payloads)):
            raise ValueError("ids, embeddings, and payloads must have the same length")

        points = [
            PointStruct(
                id=id_,
                vector=(
                    embedding.tolist()
                    if isinstance(embedding, np.ndarray)
                    else embedding
                ),
                payload=payload,
            )
            for id_, embedding, payload in zip(ids, embeddings, payloads)
        ]

        self.client.upsert(collection_name=self.collection_name, points=points)

        print(f"Added {len(points)} documents to collection '{self.collection_name}'")
        return True

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
        filter_conditions: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_conditions: Optional metadata filters (e.g., {"document_name": "contract.pdf"})

        Returns:
            List of search results with scores and metadata
        """
        # Build filter if provided
        query_filter = None
        if filter_conditions:
            must_conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_conditions.items()
            ]
            query_filter = Filter(must=must_conditions)

        # Perform search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=(
                query_vector.tolist()
                if isinstance(query_vector, np.ndarray)
                else query_vector
            ),
            limit=top_k,
            query_filter=query_filter,
            search_params=SearchParams(hnsw_ef=128),
        )

        # Format results
        results = []
        for hit in search_result:
            result = {"id": hit.id, "score": hit.score, **hit.payload}
            results.append(result)

        return results

    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if successful
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=ids),
        )
        print(f"Deleted {len(ids)} documents from collection '{self.collection_name}'")
        return True

    def get_document_count(self) -> int:
        """Get the number of documents in the collection."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count

    def collection_exists(self) -> bool:
        """Check if the collection exists."""
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        return self.collection_name in collection_names

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        print(f"Collection '{self.collection_name}' deleted.")

    def get_collection_info(self) -> dict:
        """Get information about the collection."""
        if not self.collection_exists():
            return {"exists": False}

        collection_info = self.client.get_collection(self.collection_name)
        return {
            "exists": True,
            "name": self.collection_name,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count,
            "status": collection_info.status,
        }


def main():
    """Test the vector store."""
    from src.embeddings.embedder import Embedder

    print("=== Testing Vector Store ===\n")

    # Initialize embedder and vector store
    embedder = Embedder()
    vector_store = VectorStore(embedding_dim=embedder.get_embedding_dimension())

    # Test documents
    documents = [
        {
            "id": "doc1",
            "text": "This contract specifies payment terms of 30 days.",
            "metadata": {"document_name": "contract_001.pdf", "section": "payment"},
        },
        {
            "id": "doc2",
            "text": "The agreement is governed by California law.",
            "metadata": {"document_name": "contract_001.pdf", "section": "legal"},
        },
        {
            "id": "doc3",
            "text": "Termination requires 60 days written notice.",
            "metadata": {"document_name": "contract_002.pdf", "section": "termination"},
        },
    ]

    # Generate embeddings
    print("Generating embeddings...")
    texts = [doc["text"] for doc in documents]
    embeddings = embedder.embed_batch(texts)

    # Add to vector store
    print("\nAdding documents to vector store...")
    ids = [doc["id"] for doc in documents]
    payloads = [{"text": doc["text"], **doc["metadata"]} for doc in documents]
    vector_store.add_documents(ids, embeddings, payloads)

    # Get collection info
    print("\n=== Collection Info ===")
    info = vector_store.get_collection_info()
    print(f"Points count: {info['points_count']}")
    print(f"Status: {info['status']}")

    # Test search
    print("\n=== Testing Search ===")
    query = "What are the payment requirements?"
    print(f"Query: {query}")
    query_embedding = embedder.embed_text(query)
    results = vector_store.search(query_embedding, top_k=2)

    print(f"\nTop {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Text: {result['text']}")
        print(f"   Document: {result['document_name']}")
        print(f"   Section: {result['section']}")

    # Test filtered search
    print("\n=== Testing Filtered Search ===")
    print("Filter: document_name='contract_001.pdf'")
    results = vector_store.search(
        query_embedding,
        top_k=5,
        filter_conditions={"document_name": "contract_001.pdf"},
    )
    print(f"Found {len(results)} results")


if __name__ == "__main__":
    main()
