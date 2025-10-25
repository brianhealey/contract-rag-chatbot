"""
Contract RAG chatbot integrating all components for Q&A over legal documents.
"""

from typing import List, Dict, Any, Optional
import ollama

from src.embeddings.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.reranker import Reranker
from src.retrieval.query_cache import QueryCache
from config.settings import settings


class ContractChatbot:
    """
    RAG-based chatbot for answering questions about contracts and legal documents.

    Combines hybrid search, re-ranking, and LLM generation for accurate responses.
    """

    def __init__(
        self,
        embedder: Embedder = None,
        vector_store: VectorStore = None,
        use_reranking: bool = None,
        use_cache: bool = None
    ):
        """
        Initialize the chatbot.

        Args:
            embedder: Embedder instance
            vector_store: VectorStore instance
            use_reranking: Whether to use re-ranking
            use_cache: Whether to use query caching
        """
        print("Initializing Contract RAG Chatbot...")

        # Initialize components
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore(
            embedding_dim=self.embedder.get_embedding_dimension()
        )

        self.hybrid_search = HybridSearch(
            embedder=self.embedder,
            vector_store=self.vector_store
        )

        # Load BM25 index if it exists
        if not self.hybrid_search.load_bm25_index():
            print("Warning: BM25 index not found. Only vector search will be used.")
            print("Run 'python build_index.py' to build the complete index.")

        self.use_reranking = (
            use_reranking if use_reranking is not None
            else settings.retrieval.use_reranking
        )

        if self.use_reranking:
            self.reranker = Reranker()
        else:
            self.reranker = None

        self.use_cache = (
            use_cache if use_cache is not None
            else settings.cache.enable_query_cache
        )

        if self.use_cache:
            self.query_cache = QueryCache()
        else:
            self.query_cache = None

        # Ollama configuration
        self.ollama_base_url = settings.ollama.base_url
        self.ollama_model = settings.ollama.model
        self.temperature = settings.ollama.temperature

        print(f"Chatbot initialized with model: {self.ollama_model}")
        print(f"Re-ranking: {'Enabled' if self.use_reranking else 'Disabled'}")
        print(f"Query caching: {'Enabled' if self.use_cache else 'Disabled'}")

    def retrieve_context(
        self,
        query: str,
        top_k: int = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query.

        Args:
            query: User query
            top_k: Number of results to retrieve
            filter_conditions: Optional metadata filters

        Returns:
            List of relevant passages
        """
        top_k = top_k or settings.retrieval.top_k

        # Check cache first
        cache_params = {
            "top_k": top_k,
            "use_reranking": self.use_reranking,
            "filters": filter_conditions
        }

        if self.query_cache:
            cached_results = self.query_cache.get(query, cache_params)
            if cached_results is not None:
                return cached_results

        # Perform hybrid search
        results = self.hybrid_search.hybrid_search(
            query=query,
            top_k=top_k,
            filter_conditions=filter_conditions
        )

        # Re-rank if enabled
        if self.use_reranking and self.reranker and results:
            rerank_k = settings.retrieval.rerank_top_k
            results = self.reranker.rerank(query, results, top_k=rerank_k)

        # Cache results
        if self.query_cache:
            self.query_cache.set(query, results, cache_params)

        return results

    def _format_context(self, passages: List[Dict[str, Any]]) -> str:
        """
        Format retrieved passages into context string.

        Args:
            passages: List of retrieved passages

        Returns:
            Formatted context string
        """
        if not passages:
            return "No relevant information found."

        context_parts = []
        for i, passage in enumerate(passages, 1):
            text = passage.get("text", "")
            source = passage.get("source_file", "Unknown")
            chunk_idx = passage.get("chunk_index", "")

            context_parts.append(
                f"[{i}] (Source: {source}, Chunk: {chunk_idx})\n{text}\n"
            )

        return "\n".join(context_parts)

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM."""
        return """You are a legal document assistant specialized in analyzing contracts and legal documents.

Your role is to:
- Answer questions accurately based ONLY on the provided contract passages
- Cite specific passages using [1], [2], etc. when making claims
- Clearly state when information is not available in the provided context
- Explain legal terms and clauses in clear language
- Be precise and avoid making assumptions beyond what's stated in the documents

If a question cannot be answered with the provided context, say so clearly."""

    def _build_user_prompt(self, query: str, context: str) -> str:
        """
        Build the user prompt with query and context.

        Args:
            query: User query
            context: Retrieved context passages

        Returns:
            Formatted user prompt
        """
        return f"""Context from contract documents:

{context}

Question: {query}

Please answer the question based on the context above. Cite specific passages using [1], [2], etc."""

    def generate_response(
        self,
        query: str,
        passages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a response using the LLM.

        Args:
            query: User query
            passages: Retrieved context passages

        Returns:
            Response dictionary with answer and metadata
        """
        # Format context
        context = self._format_context(passages)

        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, context)

        # Call Ollama
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.temperature
                }
            )

            answer = response["message"]["content"]

            return {
                "answer": answer,
                "passages": passages,
                "num_passages": len(passages),
                "query": query
            }

        except Exception as e:
            return {
                "answer": f"Error generating response: {str(e)}",
                "passages": passages,
                "num_passages": len(passages),
                "query": query,
                "error": str(e)
            }

    def ask(
        self,
        query: str,
        top_k: int = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        return_passages: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer with context.

        Args:
            query: User question
            top_k: Number of passages to retrieve
            filter_conditions: Optional metadata filters
            return_passages: Whether to include passages in response

        Returns:
            Response dictionary with answer and metadata
        """
        # Retrieve context
        passages = self.retrieve_context(query, top_k, filter_conditions)

        # Generate response
        response = self.generate_response(query, passages)

        # Optionally exclude passages from response
        if not return_passages:
            response.pop("passages", None)

        return response


def main():
    """Test the chatbot."""
    print("=== Testing Contract Chatbot ===\n")

    try:
        # Initialize chatbot
        chatbot = ContractChatbot()

        # Get collection info
        info = chatbot.vector_store.get_collection_info()
        print(f"\nVector store status:")
        print(f"  Collection exists: {info.get('exists', False)}")
        print(f"  Document count: {info.get('points_count', 0)}")

        if not info.get('exists') or info.get('points_count', 0) == 0:
            print("\nNo documents indexed. Please run build_index.py first.")
            return

        # Test query
        query = "What are the payment terms?"
        print(f"\n=== Query: {query} ===\n")

        response = chatbot.ask(query, top_k=5)

        print("Answer:")
        print(response["answer"])

        print(f"\n\nBased on {response['num_passages']} retrieved passages")

        if response.get("passages"):
            print("\nRetrieved passages:")
            for i, passage in enumerate(response["passages"], 1):
                score_info = []
                if "rerank_score" in passage:
                    score_info.append(f"Rerank: {passage['rerank_score']:.3f}")
                if "hybrid_score" in passage:
                    score_info.append(f"Hybrid: {passage['hybrid_score']:.3f}")

                score_str = ", ".join(score_info) if score_info else "N/A"

                print(f"\n[{i}] ({score_str})")
                print(f"Source: {passage.get('source_file', 'Unknown')}")
                print(f"Text: {passage.get('text', '')[:150]}...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
