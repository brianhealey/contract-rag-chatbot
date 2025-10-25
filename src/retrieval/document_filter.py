"""
Document filtering and comparison utilities.
"""

from typing import List, Dict, Any

from src.retrieval.vector_store import VectorStore


class DocumentFilter:
    """
    Utilities for filtering and comparing documents.
    """

    def __init__(self, vector_store: VectorStore = None):
        """
        Initialize document filter.

        Args:
            vector_store: VectorStore instance
        """
        self.vector_store = vector_store or VectorStore()

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all unique documents in the vector store.

        Returns:
            List of document info dictionaries
        """
        try:
            # Scroll through all points to get unique documents
            offset = None
            documents = {}

            while True:
                result = self.vector_store.client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in result[0]:
                    payload = point.payload
                    source_file = payload.get("source_file", "Unknown")
                    document_id = payload.get("document_id", "Unknown")

                    if document_id not in documents:
                        documents[document_id] = {
                            "document_id": document_id,
                            "source_file": source_file,
                            "file_path": payload.get("file_path", ""),
                            "file_type": payload.get("file_type", ""),
                            "file_size": payload.get("file_size", 0),
                            "total_chunks": payload.get("total_chunks", 0),
                            "parsed_at": payload.get("parsed_at", ""),
                        }

                offset = result[1]
                if offset is None:
                    break

            return list(documents.values())

        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    def get_document_chunks(
        self, document_id: str = None, source_file: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks for a specific document.

        Args:
            document_id: Document ID to filter by
            source_file: Source filename to filter by (alternative to document_id)

        Returns:
            List of chunks from the document
        """
        if not document_id and not source_file:
            raise ValueError("Must provide either document_id or source_file")

        try:
            offset = None
            chunks = []

            while True:
                result = self.vector_store.client.scroll(
                    collection_name=self.vector_store.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                for point in result[0]:
                    payload = point.payload
                    match = False

                    if document_id and payload.get("document_id") == document_id:
                        match = True
                    elif source_file and payload.get("source_file") == source_file:
                        match = True

                    if match:
                        chunks.append(
                            {
                                "id": point.id,
                                "chunk_index": payload.get("chunk_index", 0),
                                "text": payload.get("text", ""),
                                "source_file": payload.get("source_file", ""),
                                "document_id": payload.get("document_id", ""),
                                **payload,
                            }
                        )

                offset = result[1]
                if offset is None:
                    break

            # Sort by chunk index
            chunks.sort(key=lambda x: x.get("chunk_index", 0))
            return chunks

        except Exception as e:
            print(f"Error getting document chunks: {e}")
            return []

    def create_filter_condition(
        self, source_file: str = None, document_id: str = None, file_type: str = None
    ) -> Dict[str, Any]:
        """
        Create a filter condition for Qdrant search.

        Args:
            source_file: Filter by source filename
            document_id: Filter by document ID
            file_type: Filter by file type

        Returns:
            Filter condition dictionary
        """
        conditions = {}

        if source_file:
            conditions["source_file"] = source_file
        if document_id:
            conditions["document_id"] = document_id
        if file_type:
            conditions["file_type"] = file_type

        return conditions if conditions else None

    def compare_documents(
        self,
        doc1_filter: Dict[str, Any],
        doc2_filter: Dict[str, Any],
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compare two documents by retrieving relevant passages from each.

        Args:
            doc1_filter: Filter for first document
            doc2_filter: Filter for second document
            query: Query to search for in both documents
            top_k: Number of passages to retrieve from each

        Returns:
            Comparison results with passages from both documents
        """
        from src.embeddings.embedder import Embedder

        embedder = Embedder()
        query_embedding = embedder.embed_text(query)

        # Search in first document
        results1 = self.vector_store.search(
            query_vector=query_embedding, top_k=top_k, filter_conditions=doc1_filter
        )

        # Search in second document
        results2 = self.vector_store.search(
            query_vector=query_embedding, top_k=top_k, filter_conditions=doc2_filter
        )

        return {
            "query": query,
            "document1": {
                "filter": doc1_filter,
                "results": results1,
                "count": len(results1),
            },
            "document2": {
                "filter": doc2_filter,
                "results": results2,
                "count": len(results2),
            },
        }


def main():
    """Test document filtering."""
    print("=== Testing Document Filter ===\n")

    filter_util = DocumentFilter()

    # List all documents
    print("--- All Documents in Vector Store ---")
    documents = filter_util.list_documents()
    print(f"\nFound {len(documents)} document(s):\n")

    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc['source_file']}")
        print(f"   Document ID: {doc['document_id']}")
        print(f"   Type: {doc['file_type']}")
        print(f"   Size: {doc['file_size']} bytes")
        print(f"   Chunks: {doc['total_chunks']}")
        print()

    if documents:
        # Test getting chunks from first document
        first_doc = documents[0]
        print(f"\n--- Chunks from: {first_doc['source_file']} ---")
        chunks = filter_util.get_document_chunks(document_id=first_doc["document_id"])
        print(f"\nRetrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\nChunk {chunk['chunk_index']}:")
            print(f"  {chunk['text'][:100]}...")

        # Test filter creation
        print("\n--- Filter Conditions ---")
        filter_cond = filter_util.create_filter_condition(
            source_file=first_doc["source_file"]
        )
        print(f"Filter for '{first_doc['source_file']}':")
        print(f"  {filter_cond}")


if __name__ == "__main__":
    main()
