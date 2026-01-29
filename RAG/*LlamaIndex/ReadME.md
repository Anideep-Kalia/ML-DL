# LlamaIndex – Rapid Recall Notes

## One‑line Definition
**LlamaIndex** = data ingestion, indexing, and retrieval layer for LLM applications.

> Mental role: *retrieval brain* for LLM systems.

---

## Core Data Model (must‑remember)

### Nodes (most important concept)
- Nodes = **chunks of documents** (atomic retrieval unit)
- Why nodes exist:
  - LLM context limits
  - Retrieval must be granular

If you don’t understand nodes, you don’t understand LlamaIndex.

---

## Indexes (how nodes are organized)

| Index        | Retrieval logic        | Best for            | Weak at        |
| ------------ | ---------------------- | ------------------- | -------------- |
| VectorIndex  | Semantic similarity    | User Q&A, concepts  | Exact terms    |
| KeywordIndex | Exact match            | Errors, APIs, logs  | Meaning        |
| TreeIndex    | Hierarchical summaries | Multi-doc reasoning | Simple queries |
| ListIndex    | Sequential scan        | SOPs, timelines     | Large corpora  |


---

## Vector Stores (optional, common)
Pluggable backends (LlamaIndex abstracts them):
- FAISS
- Pinecone
- Weaviate
- Chroma

```python
VectorStoreIndex.from_vector_store(...)
```

Vector DB ≠ retrieval framework — LlamaIndex adds chunking, metadata, synthesis.

---

## Retrievers
- Fetch relevant nodes
- Control recall vs precision

```python
retriever = index.as_retriever(similarity_top_k=5)
```

Retriever output = context fed to LLM.

---

## Query Engine
**Query Engine = Retriever + Prompting + Response Synthesis**

```python
query_engine = index.as_query_engine()
response = query_engine.query("What is LCEL?")
```

High‑level abstraction for standard RAG.

---

## Response Synthesizers (CRITICAL)
Response synthesizers decide **how retrieved nodes are combined**.

> Bad synthesis → hallucinations  
> Good synthesis → grounded answers

### Modes (memorize)
- `compact`
- `tree_summarize`
- `refine`
- `accumulate`

### Hallucination Risk (low → high)
```
refine < tree_summarize < compact < accumulate
```

---

### `compact` (default, fastest, riskiest)
- Concatenates all nodes
- Single prompt, single answer

Use when:
- Short docs
- High‑quality retrieval

Avoid when:
- Legal / compliance / medical

---

### `tree_summarize` (best for multi‑doc reasoning)
- Group nodes → summarize → recursively summarize
- Prevents context flooding

Use when:
- Many documents
- Cross‑document synthesis
- Research‑style queries

---

### `refine` (lowest hallucination)
- Answer from first node
- Iteratively refine with next nodes

Use when:
- High accuracy required
- Sensitive or factual domains

Tradeoff: slow + token heavy

---

### `accumulate` (raw, unsafe for users)
- Answer per node
- No synthesis

Use only for:
- Debugging retrieval
- Auditing answers

Never expose directly to end users.

---

## Advanced Features (where LlamaIndex shines)

### Metadata‑aware Filtering
Prevents irrelevant context.

```python
retriever = index.as_retriever(filters={"source": "manual.pdf"})
```

Key effect: **precision control + hallucination reduction**.

---

### Hybrid Search
Combine:
- keyword + vector
- structured + unstructured

Use when:
- Error codes
- API names
- Legal clauses
- Versioned docs

Enterprise RAG requires hybrid search.

---

### Multi‑document Reasoning
- TreeIndex + `tree_summarize`
- Hierarchical aggregation
- Prevents token explosion

This is document intelligence, not basic RAG.

---

### Structured + Unstructured Retrieval
- SQL tables + PDFs + docs
- Unified answers from heterogeneous data

LlamaIndex handles this cleanly; orchestration tools struggle here.

---

### SQL / Structured Data Querying (NL → SQL)
```python
response = query_engine.query(
    "What were the top 5 selling products last month?"
)
```

Benefits:
- Deterministic results
- No embeddings needed
- Near‑zero hallucination (schema‑bound)

---

### Observability Hooks
Track:
- Retrieved nodes
- Token usage
- Prompt structure
- Answer provenance

```python
response.source_nodes
response.metadata
```

If you can’t explain *why* an answer was produced, it’s not production‑ready.

---

## Final Mental Model

- Nodes = atomic retrieval units
- Index = how nodes are organized
- Retriever = what nodes are fetched
- Synthesizer = how answers are formed

> **LlamaIndex = retrieval intelligence layer for LLM systems**

