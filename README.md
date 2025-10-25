# Contract RAG System

A production-ready Retrieval-Augmented Generation (RAG) system for querying contracts and legal documents. Built with hybrid search, re-ranking, and local LLM inference using Ollama.

## Features

- **Multi-Format Support**: PDF, HTML, TXT, DOCX, Markdown
- **Hybrid Search**: Combines vector similarity (semantic) and BM25 (keyword) search
- **Cross-Encoder Re-ranking**: Improves relevance of retrieved passages
- **Multi-Level Caching**: Embedding cache and query result cache for fast responses
- **Incremental Indexing**: Add new documents without rebuilding entire index
- **Local LLM**: Uses Ollama for privacy-preserving inference
- **Vector Database**: Qdrant for scalable vector search with HNSW indexing

## Architecture

```
┌──────────────┐
│ User Query   │
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│ Query Cache Check  │─────► Cache Hit → Return
└──────┬─────────────┘
       │ Cache Miss
       ▼
┌──────────────────────────┐
│ Embedding (384-dim)      │
│ Sentence-Transformers    │
└──────┬───────────────────┘
       │
   ┌───┴────────────────┐
   │                    │
   ▼                    ▼
┌────────────┐    ┌─────────────┐
│   Vector   │    │    BM25     │
│   Search   │    │   Search    │
│  (Qdrant)  │    │ (Keyword)   │
└──────┬─────┘    └──────┬──────┘
       │                  │
       └────────┬─────────┘
                │
                ▼
      ┌──────────────────┐
      │ Hybrid Fusion    │
      │  (α=0.5)         │
      └────────┬─────────┘
               │
               ▼
      ┌─────────────────┐
      │ Cross-Encoder   │
      │   Re-ranking    │
      └────────┬────────┘
               │
               ▼
      ┌──────────────────┐
      │ LLM Generation   │
      │    (Ollama)      │
      └────────┬─────────┘
               │
               ▼
      ┌──────────────┐
      │   Response   │
      └──────────────┘
```

## Installation

### Prerequisites

- Python 3.9+ (tested on 3.13)
- Docker (for Qdrant)
- Ollama (for LLM)

### 1. Clone and Setup

```bash
cd contract-rag

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Start Qdrant

```bash
docker-compose up -d
```

This starts Qdrant on:
- HTTP API: `http://localhost:6333`
- gRPC API: `http://localhost:6334`

### 3. Install and Start Ollama

```bash
# Install Ollama (see https://ollama.ai)
# On macOS:
brew install ollama

# Start Ollama
ollama serve

# Pull a model (in another terminal)
ollama pull llama3.2
```

### 4. Configure (Optional)

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

## Quick Start

### 1. Add Documents

Place your contract documents in the `documents/` directory:

```bash
mkdir -p documents
cp /path/to/your/contracts/*.pdf documents/
```

Supported formats: PDF, HTML, TXT, DOCX, MD

### 2. Build Index

```bash
python build_index.py
```

This will:
- Parse all documents
- Chunk text into passages
- Generate embeddings
- Build vector and BM25 indices

Options:
```bash
python build_index.py --rebuild      # Rebuild from scratch
python build_index.py --recursive    # Search subdirectories
```

### 3. Start Chatbot

```bash
python main.py
```

Interactive mode example:
```
Question: What are the payment terms?
Answer: According to the contract, payment is due within 30 days...

Question: What happens if payment is late?
Answer: Late payments incur interest at 1.5% per month...
```

## Usage

### Interactive Mode

```bash
python main.py
```

Commands:
- `help` - Show available commands
- `info` - Show system information
- `clear` - Clear query cache
- `exit` - Exit chatbot

### Single Query Mode

```bash
python main.py -q "What are the termination clauses?"
```

Options:
```bash
--no-sources    # Don't show source passages
--no-rerank     # Disable re-ranking
--no-cache      # Disable query caching
```

### Adding New Documents

Add documents incrementally without rebuilding:

```bash
python add_documents.py documents/new_contract.pdf documents/addendum.docx
```

This updates both vector and BM25 indices.

## Configuration

Edit `.env` or use environment variables:

### Qdrant Settings
```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=contract_documents
```

### Embedding Settings
```bash
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
EMBEDDING_BATCH_SIZE=32
```

### Chunking Settings
```bash
CHUNKING_CHUNK_SIZE=1000
CHUNKING_CHUNK_OVERLAP=200
```

### Retrieval Settings
```bash
RETRIEVAL_TOP_K=20              # Initial retrieval
RETRIEVAL_RERANK_TOP_K=5        # After re-ranking
RETRIEVAL_HYBRID_ALPHA=0.5      # 0=BM25 only, 1=vector only
RETRIEVAL_USE_RERANKING=true
```

### Ollama Settings
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TEMPERATURE=0.1
```

### Cache Settings
```bash
CACHE_ENABLE_QUERY_CACHE=true
CACHE_QUERY_CACHE_TTL=3600      # 1 hour
CACHE_CACHE_DIR=./cache
```

## Project Structure

```
contract-rag/
├── src/
│   ├── parsers/
│   │   ├── document_parser.py    # Multi-format document parsing
│   │   └── chunker.py            # Recursive text chunking
│   ├── embeddings/
│   │   └── embedder.py           # Sentence-Transformers + caching
│   ├── retrieval/
│   │   ├── vector_store.py       # Qdrant integration
│   │   ├── hybrid_search.py      # Vector + BM25 fusion
│   │   ├── reranker.py           # Cross-encoder re-ranking
│   │   └── query_cache.py        # Query result caching
│   └── chatbot/
│       └── contract_chatbot.py   # Main RAG orchestration
├── config/
│   └── settings.py               # Pydantic configuration
├── documents/                    # Your documents go here
├── data/                         # Generated indices
├── cache/                        # Disk cache
├── build_index.py                # Index building script
├── add_documents.py              # Incremental document addition
├── main.py                       # Interactive chatbot
├── docker-compose.yml            # Qdrant setup
├── requirements.txt
└── README.md
```

## Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | Qdrant 1.11.3 | Scalable vector search with HNSW |
| **Embeddings** | Sentence-Transformers | all-MiniLM-L6-v2 (384-dim) |
| **Re-ranking** | Cross-Encoder | ms-marco-MiniLM-L-6-v2 |
| **Keyword Search** | BM25 | Okapi BM25 for hybrid search |
| **LLM** | Ollama | Local LLM inference |
| **Document Parsing** | Unstructured | Multi-format support |
| **Caching** | DiskCache | Multi-level disk caching |
| **Config** | Pydantic | Type-safe configuration |

## Performance Tips

1. **Adjust chunk size**: Smaller chunks (500-800) for precise answers, larger (1500-2000) for context
2. **Tune hybrid alpha**: Lower (0.3-0.4) for keyword-heavy queries, higher (0.6-0.7) for semantic
3. **Re-ranking**: Disable for faster responses, enable for better accuracy
4. **Caching**: Keep enabled for repeated queries
5. **Batch size**: Increase for faster indexing on GPU

## Troubleshooting

### Qdrant Connection Error
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose restart
```

### Ollama Connection Error
```bash
# Check if Ollama is running
curl http://localhost:11434

# Start Ollama
ollama serve
```

### Out of Memory
```bash
# Reduce batch size in .env
EMBEDDING_BATCH_SIZE=16

# Use CPU instead of GPU
EMBEDDING_DEVICE=cpu
```

### Slow Performance
```bash
# Enable caching
CACHE_ENABLE_QUERY_CACHE=true

# Disable re-ranking
RETRIEVAL_USE_RERANKING=false

# Reduce retrieval count
RETRIEVAL_TOP_K=10
```

## Development

### Running Tests

```bash
# Test individual components
python -m src.parsers.document_parser
python -m src.embeddings.embedder
python -m src.retrieval.vector_store
```

### Clearing Data

```bash
# Clear cache
rm -rf cache/*

# Clear vector store (requires rebuild)
docker-compose down -v
docker-compose up -d
```

## License

MIT

## Acknowledgments

- Uses open-source models from Sentence-Transformers
- Powered by Qdrant vector database
- LLM inference via Ollama
