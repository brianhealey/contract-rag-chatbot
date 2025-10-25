# Enhanced Features

This document describes the enhanced query and document filtering capabilities added to the Contract RAG system.

## Overview of Enhancements

1. **Query Enhancement** - Improve retrieval by generating multiple query variations
2. **Document Filtering** - Search within specific documents or compare documents
3. **Document Management** - List and inspect indexed documents

## 1. Query Enhancement

Query enhancement improves retrieval by expanding or rephrasing your questions to find more relevant content.

### Strategies Available

#### Multi-Query Generation
Generates multiple variations of your question to cast a wider search net.

```bash
python main_enhanced.py --enhance
```

**Example:**
```
Original: "What are the payment terms?"
Generated:
- What are the payment terms?
- Payment schedules
- Terms of remuneration
```

#### Query Expansion
Adds related legal terms and synonyms to your query.

```python
from src.retrieval.query_enhancer import QueryEnhancer

enhancer = QueryEnhancer()
expanded = enhancer.expand_query("What are the payment terms?")
# Returns: "What are the payment terms including net payment, due date,
#           late fees, payment schedules, and payment conditions?"
```

#### HyDE (Hypothetical Document Embeddings)
Generates a hypothetical answer in contract language, then searches for similar passages.

```python
hyde_doc = enhancer.generate_hypothetical_answer("What are the payment terms?")
# Returns: "The parties agree to settle all outstanding invoices within
#           thirty (30) days of receipt of payment notice..."
```

### Using Query Enhancement

**Interactive Mode:**
```bash
# Start with enhancement enabled
python main_enhanced.py --enhance

# Or toggle during session
Question: enhance on
Question: What are the confidentiality obligations?
```

**Single Query Mode:**
```bash
python main_enhanced.py -q "What are the termination clauses?" --enhance
```

### How It Works

1. Your query is sent to the LLM to generate variations
2. Each variation is embedded and searched independently
3. Results are merged and deduplicated
4. Top results are re-ranked for relevance
5. Final answer synthesizes information from all results

## 2. Document Filtering

Filter search results to specific documents or compare content across documents.

### List All Documents

View all indexed documents:

```bash
# Command line
python main_enhanced.py --list-docs

# Interactive mode
Question: docs
```

**Output:**
```
======================================================================
Available Documents
======================================================================

Found 2 document(s):

1. service_agreement_2024.pdf
   Type: PDF
   Size: 435,782 bytes
   Chunks: 87
   ID: abc123...

2. employment_contract.pdf
   Type: PDF
   Size: 234,512 bytes
   Chunks: 45
   ID: def456...
======================================================================
```

### Filter to Specific Document

Search within a single document:

```bash
# Command line - single query
python main_enhanced.py -q "What are the payment terms?" \
  --filter "service_agreement_2024.pdf"

# Interactive mode
Question: filter service_agreement_2024.pdf
Question: What are the payment terms?
Question: unfilter  # Remove filter
```

**Use Cases:**
- Compare clauses across different versions
- Focus search on a specific contract type
- Verify information is from the correct document

### Compare Two Documents

Compare how two documents address the same topic:

```bash
# Interactive mode only
Question: compare
```

**Interactive Workflow:**
1. Lists all available documents
2. Select first document by number
3. Select second document by number
4. Enter comparison query
5. View side-by-side results

**Example Output:**
```
======================================================================
Results for: termination conditions
======================================================================

--- service_agreement_2024.pdf ---

[1] Score: 0.8234 | Chunk 23
    Either party may terminate this Agreement upon 30 days written
    notice for any reason...

[2] Score: 0.7891 | Chunk 45
    Immediate termination is permitted in case of material breach...

--- employment_contract.pdf ---

[1] Score: 0.8567 | Chunk 12
    The Company retains the right to terminate employment at any time,
    with or without cause...

[2] Score: 0.7234 | Chunk 34
    Employee may resign by providing 14 days written notice...
======================================================================
```

## 3. Document Management API

Programmatically work with documents in Python:

### List Documents

```python
from src.retrieval.document_filter import DocumentFilter

doc_filter = DocumentFilter()
documents = doc_filter.list_documents()

for doc in documents:
    print(f"{doc['source_file']}: {doc['total_chunks']} chunks")
```

### Get All Chunks from a Document

```python
# By document ID
chunks = doc_filter.get_document_chunks(document_id="abc123...")

# By filename
chunks = doc_filter.get_document_chunks(
    source_file="service_agreement_2024.pdf"
)

# Inspect chunks
for chunk in chunks[:5]:
    print(f"Chunk {chunk['chunk_index']}: {chunk['text'][:100]}...")
```

### Create Search Filters

```python
# Filter by source file
filter_cond = doc_filter.create_filter_condition(
    source_file="employment_contract.pdf"
)

# Filter by file type
filter_cond = doc_filter.create_filter_condition(
    file_type="pdf"
)

# Use in search
from src.chatbot.contract_chatbot import ContractChatbot

chatbot = ContractChatbot()
response = chatbot.ask(
    "What are the payment terms?",
    filter_conditions=filter_cond
)
```

### Compare Documents Programmatically

```python
comparison = doc_filter.compare_documents(
    doc1_filter={'source_file': 'contract_v1.pdf'},
    doc2_filter={'source_file': 'contract_v2.pdf'},
    query="What are the liability limitations?",
    top_k=3
)

# Access results
doc1_results = comparison['document1']['results']
doc2_results = comparison['document2']['results']
```

## Command Reference

### Enhanced Main Script

```bash
python main_enhanced.py [options]
```

**Options:**
- `-q, --question TEXT` - Ask a single question and exit
- `--enhance` - Enable query enhancement (multi-query)
- `--filter FILENAME` - Filter to specific document
- `--list-docs` - List all documents and exit
- `--no-sources` - Don't show source passages
- `--no-rerank` - Disable re-ranking
- `--no-cache` - Disable query caching

### Interactive Commands

While in interactive mode:
- `docs` - List all documents
- `filter <filename>` - Filter to specific document
- `unfilter` - Remove document filter
- `compare` - Compare two documents
- `enhance on/off` - Toggle query enhancement
- `info` - Show system information
- `clear` - Clear query cache
- `help` - Show help
- `exit` - Exit chatbot

## Examples

### Example 1: Enhanced Search Across All Documents

```bash
python main_enhanced.py --enhance
```

```
Question: What are the confidentiality obligations?
Enhancing query...
Generated 3 query variations.

Answer:
[Multiple passages synthesized from query variations...]

Sources (5 passages):
[1] service_agreement_2024.pdf (Chunk 34) | Rerank: 8.234
[2] employment_contract.pdf (Chunk 12) | Rerank: 7.891
[3] nda_agreement.pdf (Chunk 5) | Rerank: 7.456
...
```

### Example 2: Filtered Search

```bash
python main_enhanced.py
```

```
Question: filter employment_contract.pdf
Filtering to: employment_contract.pdf

Question: What is the base salary?
Answer:
[Results only from employment_contract.pdf]

Question: unfilter
Filter removed.
```

### Example 3: Document Comparison

```bash
python main_enhanced.py
```

```
Question: compare
Available documents:
1. contract_2023.pdf
2. contract_2024.pdf

Select first document: 1
Select second document: 2

Enter comparison query: payment terms

Results for: payment terms

--- contract_2023.pdf ---
[Shows payment terms from 2023 contract]

--- contract_2024.pdf ---
[Shows payment terms from 2024 contract]
```

### Example 4: Programmatic Usage

```python
from src.chatbot.contract_chatbot import ContractChatbot
from src.retrieval.query_enhancer import QueryEnhancer
from src.retrieval.document_filter import DocumentFilter

# Initialize
chatbot = ContractChatbot()
enhancer = QueryEnhancer()
doc_filter = DocumentFilter(chatbot.vector_store)

# List documents
docs = doc_filter.list_documents()
print(f"Found {len(docs)} documents")

# Enhanced query with document filter
enhanced_queries = enhancer.generate_multi_queries(
    "What are the termination conditions?",
    num_queries=3
)

filter_cond = {'source_file': docs[0]['source_file']}

# Search with enhancement
all_results = []
for query in enhanced_queries:
    results = chatbot.retrieve_context(query, filter_conditions=filter_cond)
    all_results.extend(results)

# Deduplicate and get answer
seen = {}
for r in all_results:
    if r['id'] not in seen:
        seen[r['id']] = r

passages = list(seen.values())[:5]
response = chatbot.generate_response(enhanced_queries[0], passages)
print(response['answer'])
```

## Performance Considerations

### Query Enhancement
- **Pros**: Better recall, finds related content, more comprehensive answers
- **Cons**: 2-3x slower due to multiple searches and LLM calls
- **Best for**: Complex queries, research questions, comprehensive analysis

### Document Filtering
- **Pros**: Faster searches, precise targeting, reduces noise
- **Cons**: May miss relevant info in other documents
- **Best for**: Known document scope, version comparisons, focused queries

### Recommendations

1. **Use query enhancement when:**
   - You need comprehensive information
   - Query is ambiguous or could be phrased multiple ways
   - You're exploring a topic across multiple documents

2. **Use document filtering when:**
   - You know which document contains the answer
   - Comparing specific documents
   - Working with a large document collection

3. **Combine both when:**
   - Doing detailed analysis of specific documents
   - Comparing how different documents address the same topic
   - Need both breadth (enhancement) and focus (filtering)

## Configuration

Add to your `.env` file:

```bash
# Query Enhancement
QUERY_ENHANCEMENT_ENABLED=true
QUERY_NUM_VARIATIONS=3

# Document Filtering
DOCUMENT_FILTER_ENABLED=true
```

## Troubleshooting

### Query Enhancement Not Working
- Check Ollama is running: `curl http://localhost:11434`
- Verify model is available: `ollama list`
- Check temperature settings in query_enhancer.py

### Document Filtering Returns No Results
- Verify document name with `--list-docs`
- Check for exact filename match (case-sensitive)
- Ensure documents are indexed: `python build_index.py`

### Comparison Mode Shows Wrong Documents
- List documents first: `docs`
- Note the exact filenames
- Use those exact names in comparison

## Future Enhancements

Planned features:
- [ ] Query enhancement caching
- [ ] Bulk document comparison
- [ ] Custom query templates
- [ ] Document similarity scoring
- [ ] Cross-document entity linking
- [ ] Temporal filtering (by parsed_at date)
