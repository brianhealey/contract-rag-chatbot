"""
Embedding generation module with disk caching support.
"""

import hashlib
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from diskcache import Cache
from tqdm import tqdm

from config.settings import settings


class Embedder:
    """
    Wrapper for SentenceTransformer with disk-based caching.

    Caches embeddings to avoid redundant computation for the same texts.
    """

    def __init__(
        self,
        model_name: str = None,
        cache_dir: str = None,
        device: str = None,
        batch_size: int = None
    ):
        """
        Initialize the embedder with caching.

        Args:
            model_name: Name of the sentence-transformer model
            cache_dir: Directory for disk cache
            device: Device to run model on (cpu/cuda/mps)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name or settings.embedding.model_name
        self.cache_dir = cache_dir or settings.cache.cache_dir
        self.device = device or settings.embedding.device
        self.batch_size = batch_size or settings.embedding.batch_size

        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded. Embedding dimension: {self.embedding_dim}")

        # Initialize disk cache
        self.cache = Cache(self.cache_dir)
        print(f"Cache initialized at: {self.cache_dir}")

    def _get_cache_key(self, text: str) -> str:
        """
        Generate a cache key for a given text.

        Args:
            text: Input text

        Returns:
            MD5 hash of model_name:text
        """
        key_string = f"{self.model_name}:{text}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def embed_text(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text
            use_cache: Whether to use caching

        Returns:
            Embedding vector as numpy array
        """
        if use_cache:
            cache_key = self._get_cache_key(text)
            cached_embedding = self.cache.get(cache_key)

            if cached_embedding is not None:
                return np.array(cached_embedding)

        # Compute embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Cache the result
        if use_cache:
            self.cache.set(cache_key, embedding.tolist(), expire=settings.cache.embedding_cache_ttl)

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        use_cache: bool = True,
        show_progress: bool = True
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts with caching.

        Args:
            texts: List of input texts
            use_cache: Whether to use caching
            show_progress: Show progress bar

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        if use_cache:
            for idx, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                cached_embedding = self.cache.get(cache_key)

                if cached_embedding is not None:
                    embeddings.append((idx, np.array(cached_embedding)))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(idx)

            if show_progress and len(uncached_texts) < len(texts):
                print(f"Cache hits: {len(texts) - len(uncached_texts)}/{len(texts)}")
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Compute embeddings for uncached texts
        if uncached_texts:
            if show_progress:
                print(f"Computing embeddings for {len(uncached_texts)} texts...")

            # Process in batches
            new_embeddings = []
            for i in tqdm(
                range(0, len(uncached_texts), self.batch_size),
                disable=not show_progress,
                desc="Encoding"
            ):
                batch = uncached_texts[i:i + self.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                new_embeddings.extend(batch_embeddings)

            # Cache new embeddings
            if use_cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    cache_key = self._get_cache_key(text)
                    self.cache.set(
                        cache_key,
                        embedding.tolist(),
                        expire=settings.cache.embedding_cache_ttl
                    )

            # Add to results with correct indices
            for idx, embedding in zip(uncached_indices, new_embeddings):
                embeddings.append((idx, embedding))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        return [emb for _, emb in embeddings]

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension of the model."""
        return self.embedding_dim

    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()
        print("Cache cleared.")

    def get_cache_info(self) -> dict:
        """Get information about the cache."""
        return {
            "size": len(self.cache),
            "directory": self.cache_dir,
            "model": self.model_name
        }


def main():
    """Test the embedder."""
    embedder = Embedder()

    # Test single embedding
    print("\n=== Testing single embedding ===")
    text = "This is a test contract clause about payment terms."
    embedding = embedder.embed_text(text)
    print(f"Text: {text}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")

    # Test batch embedding
    print("\n=== Testing batch embedding ===")
    texts = [
        "The party agrees to pay within 30 days.",
        "This agreement is governed by the laws of California.",
        "Either party may terminate this agreement with written notice.",
        "Confidential information must not be disclosed to third parties."
    ]
    embeddings = embedder.embed_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Test cache
    print("\n=== Testing cache (second call) ===")
    embeddings_cached = embedder.embed_batch(texts)
    print(f"Retrieved {len(embeddings_cached)} embeddings from cache")

    # Cache info
    print("\n=== Cache info ===")
    print(embedder.get_cache_info())


if __name__ == "__main__":
    main()
