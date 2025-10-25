"""
Query result caching to improve response times for repeated queries.
"""

import hashlib
from typing import List, Dict, Any, Optional
from diskcache import Cache

from config.settings import settings


class QueryCache:
    """
    Disk-based cache for query results with TTL support.
    """

    def __init__(self, cache_dir: str = None, ttl: int = None, enabled: bool = None):
        """
        Initialize the query cache.

        Args:
            cache_dir: Directory for disk cache
            ttl: Time-to-live in seconds (0 = no expiration)
            enabled: Whether caching is enabled
        """
        self.cache_dir = cache_dir or settings.cache.cache_dir
        self.ttl = ttl if ttl is not None else settings.cache.query_cache_ttl
        self.enabled = (
            enabled if enabled is not None else settings.cache.enable_query_cache
        )

        if self.enabled:
            self.cache = Cache(self.cache_dir)
            print(f"Query cache initialized at: {self.cache_dir}")
            print(
                f"Cache TTL: {self.ttl}s ({self.ttl/3600:.1f} hours)"
                if self.ttl > 0
                else "Cache TTL: No expiration"
            )
        else:
            self.cache = None
            print("Query cache disabled")

    def _generate_cache_key(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a cache key for a query and parameters.

        Args:
            query: Search query
            params: Additional search parameters

        Returns:
            MD5 hash as cache key
        """
        # Include query and relevant parameters in key
        key_string = query

        if params:
            # Sort params for consistent hashing
            sorted_params = sorted(params.items())
            key_string += ":" + str(sorted_params)

        return hashlib.md5(key_string.encode()).hexdigest()

    def get(
        self, query: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results for a query.

        Args:
            query: Search query
            params: Search parameters used

        Returns:
            Cached results if found, None otherwise
        """
        if not self.enabled:
            return None

        cache_key = self._generate_cache_key(query, params)
        cached_results = self.cache.get(cache_key)

        if cached_results is not None:
            print(f"Cache hit for query: '{query[:50]}...'")
            return cached_results

        return None

    def set(
        self,
        query: str,
        results: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Cache results for a query.

        Args:
            query: Search query
            results: Results to cache
            params: Search parameters used
        """
        if not self.enabled:
            return

        cache_key = self._generate_cache_key(query, params)
        expire_time = self.ttl if self.ttl > 0 else None

        self.cache.set(cache_key, results, expire=expire_time)

    def clear(self):
        """Clear all cached queries."""
        if self.enabled:
            self.cache.clear()
            print("Query cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        if not self.enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": len(self.cache),
            "directory": self.cache_dir,
            "ttl": self.ttl,
        }


def main():
    """Test the query cache."""
    print("=== Testing Query Cache ===\n")

    # Initialize cache with short TTL for testing
    cache = QueryCache(ttl=10)  # 10 second TTL

    # Test data
    query1 = "What are the payment terms?"
    params1 = {"top_k": 5, "use_reranking": True}
    results1 = [
        {"id": "chunk_001", "text": "Payment within 30 days", "score": 0.9},
        {"id": "chunk_002", "text": "Late fees apply", "score": 0.7},
    ]

    query2 = "What are the termination clauses?"
    results2 = [{"id": "chunk_010", "text": "60 days notice required", "score": 0.85}]

    # Test cache set
    print("Setting cache for query 1...")
    cache.set(query1, results1, params1)

    print("Setting cache for query 2...")
    cache.set(query2, results2)

    # Test cache get
    print("\n=== Testing Cache Retrieval ===\n")

    print("Retrieving query 1...")
    cached1 = cache.get(query1, params1)
    print(f"Found {len(cached1)} results" if cached1 else "No cache hit")

    print("\nRetrieving query 2...")
    cached2 = cache.get(query2)
    print(f"Found {len(cached2)} results" if cached2 else "No cache hit")

    # Test cache miss (different params)
    print("\nRetrieving query 1 with different params...")
    cached3 = cache.get(query1, {"top_k": 10, "use_reranking": False})
    print(f"Found {len(cached3)} results" if cached3 else "No cache hit (expected)")

    # Cache info
    print("\n=== Cache Info ===")
    info = cache.get_cache_info()
    print(f"Enabled: {info['enabled']}")
    print(f"Size: {info['size']} entries")
    print(f"TTL: {info['ttl']} seconds")

    # Test clear
    print("\n=== Testing Cache Clear ===")
    cache.clear()
    info = cache.get_cache_info()
    print(f"Size after clear: {info['size']} entries")


if __name__ == "__main__":
    main()
