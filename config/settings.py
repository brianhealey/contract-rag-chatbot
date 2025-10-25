"""
Configuration settings for the Contract RAG system using Pydantic.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class QdrantConfig(BaseSettings):
    """Qdrant vector database configuration."""

    host: str = Field(default="localhost", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant HTTP port")
    grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    collection_name: str = Field(default="contract_documents", description="Qdrant collection name")
    use_grpc: bool = Field(default=False, description="Use gRPC instead of HTTP")

    model_config = SettingsConfigDict(env_prefix="QDRANT_")


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformers model name"
    )
    embedding_dimension: int = Field(default=384, description="Embedding vector dimension")
    batch_size: int = Field(default=32, description="Batch size for encoding")
    device: str = Field(default="cpu", description="Device for model (cpu/cuda/mps)")

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        protected_namespaces=()
    )


class ChunkingConfig(BaseSettings):
    """Document chunking configuration."""

    chunk_size: int = Field(default=1000, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in characters")
    separators: list[str] = Field(
        default=["\n\n", "\n", ". ", ", ", " ", ""],
        description="List of separators for recursive chunking"
    )

    model_config = SettingsConfigDict(env_prefix="CHUNKING_")


class RerankConfig(BaseSettings):
    """Re-ranking model configuration."""

    model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for re-ranking"
    )
    device: str = Field(default="cpu", description="Device for model (cpu/cuda/mps)")

    model_config = SettingsConfigDict(
        env_prefix="RERANK_",
        protected_namespaces=()
    )


class RetrievalConfig(BaseSettings):
    """Retrieval configuration."""

    top_k: int = Field(default=20, description="Number of results for initial retrieval")
    rerank_top_k: int = Field(default=5, description="Number of results after re-ranking")
    hybrid_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for hybrid search (0=BM25 only, 1=vector only)"
    )
    use_reranking: bool = Field(default=True, description="Enable cross-encoder re-ranking")

    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_")


class OllamaConfig(BaseSettings):
    """Ollama LLM configuration."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    model: str = Field(default="llama3.2", description="Ollama model name")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    timeout: int = Field(default=120, description="Request timeout in seconds")

    model_config = SettingsConfigDict(env_prefix="OLLAMA_")


class CacheConfig(BaseSettings):
    """Caching configuration."""

    cache_dir: str = Field(default="./cache", description="Directory for disk cache")
    embedding_cache_ttl: int = Field(
        default=0,
        description="Embedding cache TTL in seconds (0 = no expiration)"
    )
    query_cache_ttl: int = Field(
        default=3600,
        description="Query cache TTL in seconds (default: 1 hour)"
    )
    enable_query_cache: bool = Field(default=True, description="Enable query result caching")

    model_config = SettingsConfigDict(env_prefix="CACHE_")


class DocumentConfig(BaseSettings):
    """Document processing configuration."""

    documents_dir: str = Field(default="./documents", description="Directory containing documents")
    data_dir: str = Field(default="./data", description="Directory for generated data (indices, etc.)")
    supported_formats: list[str] = Field(
        default=["pdf", "txt", "html", "docx", "md"],
        description="Supported document formats"
    )

    model_config = SettingsConfigDict(env_prefix="DOCUMENT_")


class Settings(BaseSettings):
    """Main settings container."""

    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    document: DocumentConfig = Field(default_factory=DocumentConfig)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False
    )


# Global settings instance
settings = Settings()


# Helper function to create necessary directories
def setup_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        settings.document.documents_dir,
        settings.document.data_dir,
        settings.cache.cache_dir,
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    print(f"Created directories: {', '.join(dirs)}")


if __name__ == "__main__":
    # Test configuration loading
    print("=== Contract RAG Configuration ===\n")
    print(f"Qdrant: {settings.qdrant.host}:{settings.qdrant.port}")
    print(f"Collection: {settings.qdrant.collection_name}")
    print(f"Embedding Model: {settings.embedding.model_name}")
    print(f"Chunk Size: {settings.chunking.chunk_size} (overlap: {settings.chunking.chunk_overlap})")
    print(f"Reranking: {settings.retrieval.use_reranking}")
    print(f"Ollama Model: {settings.ollama.model} @ {settings.ollama.base_url}")
    print(f"Documents Dir: {settings.document.documents_dir}")
    setup_directories()
