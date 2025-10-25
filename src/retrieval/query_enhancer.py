"""
Query enhancement for improved retrieval.

Implements multiple strategies:
- Multi-query generation: Generate alternative phrasings
- Query expansion: Add related terms and context
- HyDE: Hypothetical Document Embeddings
"""

from typing import List, Dict, Any
import ollama

from config.settings import settings


class QueryEnhancer:
    """
    Enhance user queries to improve retrieval quality.
    """

    def __init__(
        self,
        model: str = None,
        enable_multi_query: bool = True,
        enable_expansion: bool = True,
        enable_hyde: bool = False,
    ):
        """
        Initialize query enhancer.

        Args:
            model: Ollama model to use for enhancement
            enable_multi_query: Generate multiple query variations
            enable_expansion: Expand query with related terms
            enable_hyde: Use hypothetical document embeddings
        """
        self.model = model or settings.ollama.model
        self.enable_multi_query = enable_multi_query
        self.enable_expansion = enable_expansion
        self.enable_hyde = enable_hyde

    def generate_multi_queries(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple variations of a query for better coverage.

        Args:
            query: Original user query
            num_queries: Number of query variations to generate

        Returns:
            List of query variations (includes original)
        """
        prompt = f"""You are an expert at understanding legal and contract queries.
Generate {num_queries - 1} alternative phrasings of the following question that would help find relevant information in contract documents.

Original question: {query}

Requirements:
- Each variation should ask for the same information in a different way
- Use synonyms and legal terminology where appropriate
- Keep variations concise and clear
- Output ONLY the alternative questions, one per line
- Do not include explanations or numbering

Alternative questions:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.7},
            )

            variations = [
                line.strip()
                for line in response["message"]["content"].strip().split("\n")
                if line.strip()
                and not line.strip().startswith(("-", "*", "1", "2", "3", "4", "5"))
            ]

            # Return original + variations
            return [query] + variations[: num_queries - 1]

        except Exception as e:
            print(f"Warning: Multi-query generation failed: {e}")
            return [query]

    def expand_query(self, query: str) -> str:
        """
        Expand query with related terms and context.

        Args:
            query: Original user query

        Returns:
            Expanded query with additional context
        """
        prompt = f"""You are a legal document search expert.
Expand the following question by adding relevant legal terms, synonyms, and related concepts that would help find comprehensive information in contract documents.

Original question: {query}

Requirements:
- Add 3-5 related legal terms or synonyms
- Include broader and narrower terms
- Keep the expansion concise (1-2 sentences)
- Focus on search terms, not explanations
- Output ONLY the expanded query

Expanded query:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )

            expanded = response["message"]["content"].strip()
            return expanded if expanded else query

        except Exception as e:
            print(f"Warning: Query expansion failed: {e}")
            return query

    def generate_hypothetical_answer(self, query: str) -> str:
        """
        Generate a hypothetical answer (HyDE) to improve retrieval.

        Creates a hypothetical passage that would answer the query,
        which can be embedded and used for similarity search.

        Args:
            query: User query

        Returns:
            Hypothetical answer passage
        """
        prompt = f"""You are a legal document expert.
Write a hypothetical passage from a contract document that would answer the following question.

Question: {query}

Requirements:
- Write in formal contract language
- Be specific and detailed (2-3 sentences)
- Include typical legal terminology
- Do not use "hypothetical" or meta-language
- Output ONLY the passage, no explanations

Passage:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.5},
            )

            hyde_answer = response["message"]["content"].strip()
            return hyde_answer if hyde_answer else query

        except Exception as e:
            print(f"Warning: HyDE generation failed: {e}")
            return query

    def enhance_query(self, query: str, strategy: str = "auto") -> Dict[str, Any]:
        """
        Enhance a query using configured strategies.

        Args:
            query: Original user query
            strategy: Enhancement strategy ('multi', 'expand', 'hyde', 'auto')

        Returns:
            Dictionary with enhanced queries and metadata
        """
        result = {"original": query, "queries": [query], "strategy": strategy}

        if strategy == "auto":
            # Use all enabled strategies
            if self.enable_multi_query:
                result["queries"] = self.generate_multi_queries(query)
            elif self.enable_expansion:
                result["queries"] = [self.expand_query(query)]
            elif self.enable_hyde:
                result["queries"] = [self.generate_hypothetical_answer(query)]

        elif strategy == "multi" and self.enable_multi_query:
            result["queries"] = self.generate_multi_queries(query)

        elif strategy == "expand" and self.enable_expansion:
            result["queries"] = [self.expand_query(query)]

        elif strategy == "hyde" and self.enable_hyde:
            result["queries"] = [self.generate_hypothetical_answer(query)]

        return result


def main():
    """Test query enhancement."""
    print("=== Testing Query Enhancer ===\n")

    enhancer = QueryEnhancer()

    # Test queries
    test_queries = [
        "What are the payment terms?",
        "How can the contract be terminated?",
        "What are the confidentiality obligations?",
    ]

    for query in test_queries:
        print(f"\n{'=' * 70}")
        print(f"Original Query: {query}")
        print("=" * 70)

        # Test multi-query
        print("\n--- Multi-Query Variations ---")
        variations = enhancer.generate_multi_queries(query, num_queries=3)
        for i, var in enumerate(variations, 1):
            print(f"{i}. {var}")

        # Test query expansion
        print("\n--- Query Expansion ---")
        expanded = enhancer.expand_query(query)
        print(expanded)

        # Test HyDE
        print("\n--- Hypothetical Document (HyDE) ---")
        hyde = enhancer.generate_hypothetical_answer(query)
        print(hyde)


if __name__ == "__main__":
    main()
