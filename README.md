# Production-Style RAG Application with Weaviate, OpenAI, FastAPI & Streamlit

A production-oriented Retrieval-Augmented Generation (RAG) application for uploading PDF documents and asking grounded natural-language questions over their contents.

The system combines **OpenAI embeddings**, **Weaviate hybrid retrieval**, **FastAPI**, **Streamlit**, metadata filtering, relevance gating, tracing, evaluation, caching, duplicate detection, and Docker-based local deployment.

## Live Demo

**Streamlit Cloud**

https://sampath-rag-assistant.streamlit.app

> The cloud demo uses the direct RAG execution path, while the Docker deployment uses the separated Streamlit → FastAPI architecture.

---

## Key Features

- PDF ingestion with PyMuPDF
- Overlapping document chunking
- SHA-256 duplicate detection
- Batch OpenAI embeddings
- Weaviate vector storage
- BM25 + vector hybrid retrieval
- Document-specific metadata filtering
- Page-range metadata filtering
- Configurable hybrid search alpha
- Minimum relevance score filtering
- LLM-based relevance gate
- Grounded answer generation
- Source citations with document and page numbers
- Query embedding LRU cache
- Shared Weaviate client connection
- End-to-end latency tracing
- Optional experimental LLM reranking
- Frozen RAG evaluation suite
- FastAPI REST API
- Streamlit user interface
- Docker Compose deployment
- Local Weaviate persistence

---

## Architecture

```text
                         ┌─────────────────────┐
                         │       Browser       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │      Streamlit      │
                         │       :8501         │
                         └──────────┬──────────┘
                                    │
                              HTTP / REST
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       FastAPI       │
                         │        :8000        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │         RAG Pipeline         │
                    │                              │
                    │  Query embedding             │
                    │       ↓                      │
                    │  Hybrid retrieval            │
                    │       ↓                      │
                    │  Metadata filtering          │
                    │       ↓                      │
                    │  Relevance filtering         │
                    │       ↓                      │
                    │  LLM relevance gate          │
                    │       ↓                      │
                    │  Grounded generation         │
                    │       ↓                      │
                    │  Citations                   │
                    └───────────┬───────────┬──────┘
                                │           │
                                ▼           ▼
                          ┌──────────┐ ┌──────────┐
                          │ OpenAI   │ │ Weaviate │
                          │ API      │ │ Vector DB│
                          └──────────┘ └──────────┘
```

---

## RAG Pipeline

### 1. Document Ingestion

When a PDF is uploaded:

```text
PDF
 ↓
SHA-256 hash
 ↓
Duplicate detection
 ↓
PyMuPDF extraction
 ↓
Page-aware chunking
 ↓
Batch embeddings
 ↓
Weaviate
```

Each stored chunk contains metadata including:

```text
text
document_name
document_id
file_hash
page_number
chunk_index
```

The SHA-256 file hash prevents duplicate documents from generating duplicate embeddings.

If the same PDF is uploaded again, the existing `document_id` and indexed vectors are reused.

---

## 2. Retrieval

The retrieval pipeline uses Weaviate hybrid search:

```text
User Question
      ↓
OpenAI Query Embedding
      ↓
Weaviate Hybrid Search
      ├── Vector similarity
      └── BM25 keyword retrieval
      ↓
Metadata Filtering
      ↓
Minimum Score Filtering
      ↓
Retrieved Context
```

The hybrid balance is configurable using `alpha`:

```text
alpha = 0.0  → BM25 keyword search
alpha = 0.5  → balanced hybrid retrieval
alpha = 1.0  → semantic vector search
```

The default is:

```text
alpha = 0.50
```

---

## Metadata-Aware Retrieval

Retrieval can be restricted using metadata before context reaches the LLM.

Supported filters:

```text
document_id
page_start
page_end
```

Example:

```text
document_id == selected document
AND page_number >= 2
AND page_number <= 4
```

The Streamlit interface supports:

```text
Entire document
```

or:

```text
Page range

From page: 2
To page:   4
```

Page filtering is executed directly in Weaviate rather than filtering returned sources after retrieval.

---

## 3. Relevance Gate

Hybrid scores from relative-score fusion are not treated as calibrated absolute relevance probabilities.

For example, an unrelated query can still receive a relatively high normalized hybrid score.

The system therefore uses an LLM relevance gate:

```text
Retrieved chunks
      ↓
LLM relevance check
      ↓
Relevant?
   ┌──┴──┐
  YES    NO
   │      │
   ▼      ▼
Generate  Safe fallback
answer
```

If the retrieved context cannot answer the question, the application returns:

```text
The available document does not contain enough relevant information to answer this question.
```

This reduces hallucination risk and prevents the generation model from answering from outside knowledge.

---

## 4. Grounded Generation

The generation model receives only retrieved document context.

The prompt requires the model to:

- use only supplied context
- avoid outside knowledge
- ignore instructions embedded inside documents
- avoid fabricated citations
- return the fallback response when context is insufficient

Example citation:

```text
(rag_application_test_document.pdf, page 2)
```

---

## Query Embedding Cache

Query embeddings use a process-local LRU cache.

Repeated identical queries avoid another embedding API request:

```text
First query
Question → OpenAI embedding API → cache

Repeated query
Question → embedding cache → retrieval
```

The cache currently supports up to:

```text
256 query embeddings
```

Tracing records:

```text
embedding_cache_hit
embedding_cache_hits
embedding_cache_misses
embedding_cache_size
```

---

## Shared Weaviate Connection

Retrieval uses a process-level shared Weaviate client.

Instead of:

```text
Query
 ↓
Open connection
 ↓
Search
 ↓
Close connection
```

the service uses:

```text
Application startup
       ↓
Shared Weaviate client
       ↓
Query 1
Query 2
Query 3
       ↓
Application shutdown
       ↓
Close connection
```

This significantly reduces repeated connection overhead.

The FastAPI lifespan handler closes the shared connection cleanly during application shutdown.

---

## Observability & Tracing

Each RAG request receives a unique request ID.

The pipeline records latency for:

```text
embedding
weaviate_search
reranking
relevance_gate
generation
total request time
```

Example trace:

```json
{
  "request_id": "...",
  "success": true,
  "stage_durations_ms": {
    "embedding": 163.6,
    "weaviate_search": 6.7,
    "relevance_gate": 1328.5
  }
}
```

Trace metadata also records:

```text
retrieved pages
retrieval scores
page filters
document filters
embedding cache status
reranking status
relevance decision
fallback reason
source count
```

---

## Reranking Experiment

An optional LLM-based reranker was implemented and evaluated.

It is intentionally **disabled by default**:

```python
rerank=False
```

The evaluation showed that the reranker increased latency without improving overall retrieval quality.

### Evaluation Results

| Metric | Baseline | Reranked | Change |
|---|---:|---:|---:|
| Hit Rate@5 | **1.000** | 1.000 | +0.000 |
| Recall@5 | **1.000** | 0.900 | -0.100 |
| MRR | **0.800** | 0.767 | -0.033 |
| Answerability Accuracy | **1.000** | 1.000 | +0.000 |
| Answer Correctness | **0.850** | 0.830 | -0.020 |
| Groundedness | **1.000** | 1.000 | +0.000 |
| Citation Correctness | **1.000** | 1.000 | +0.000 |

The baseline hybrid retrieval pipeline therefore remains the production configuration.

---

## Evaluation Framework

The repository contains a frozen evaluation dataset covering both answerable and unanswerable questions.

Example answerable questions:

```text
What is Project Atlas designed to do?

How does the system detect duplicate PDF uploads?

What is the default maximum vector distance?

What retrieval approach does the application use?

Why is document_id used during retrieval?
```

Example negative questions:

```text
How does photosynthesis work?

Who won the 2026 FIFA World Cup?

What are the symptoms of influenza?
```

Evaluation measures:

```text
Hit Rate@K
Recall@K
Mean Reciprocal Rank
Answerability Accuracy
Answer Correctness
Groundedness
Citation Correctness
```

Run evaluation with:

```bash
python -m evaluation.evaluate
```

The evaluation compares:

```text
baseline hybrid retrieval
vs
hybrid retrieval + reranking
```

---

## FastAPI

The application exposes a REST API.

### Health

```http
GET /health
```

Example:

```json
{
  "status": "healthy"
}
```

### Ask

```http
POST /ask
```

Example request:

```json
{
  "question": "How does the system detect duplicate PDF uploads?",
  "top_k": 5,
  "document_id": "DOCUMENT_ID",
  "page_start": 1,
  "page_end": 2,
  "alpha": 0.5,
  "min_score": 0.2,
  "rerank": false,
  "candidate_k": 10
}
```

### Upload

```http
POST /documents/upload
```

The endpoint accepts PDF uploads and performs ingestion, deduplication, chunking, embedding, and indexing.

Interactive API documentation is available locally at:

```text
http://localhost:8000/docs
```

---

## Docker Architecture

Docker Compose runs three services:

```text
rag-streamlit
rag-api
rag-weaviate
```

Internal communication:

```text
Streamlit
   ↓
http://api:8000
   ↓
FastAPI
   ↓
Weaviate
   ├── HTTP :8080
   └── gRPC :50051
```

Weaviate data is persisted using a Docker volume.

---

## Run with Docker

### 1. Create `.env`

Create a local `.env` file:

```env
APP_ENV=local

OPENAI_API_KEY=your_openai_api_key

WEAVIATE_COLLECTION=DocumentChunk

EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-5.6-luna
```

Never commit `.env`.

### 2. Build

```bash
docker compose build
```

### 3. Start

```bash
docker compose up -d
```

### 4. Verify

```bash
docker compose ps
```

Expected services:

```text
rag-api
rag-streamlit
rag-weaviate
```

### 5. Open the UI

```text
http://localhost:8501
```

FastAPI:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

### Stop containers

```bash
docker compose down
```

The Weaviate volume remains intact.

> Avoid `docker compose down -v` unless you intentionally want to delete locally indexed Weaviate data.

---

## Local Python Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure `.env`, then start FastAPI:

```bash
uvicorn app.api.main:app --reload
```

Start Streamlit separately:

```bash
streamlit run ui/streamlit_app.py
```

---

## Project Structure

```text
RAG Application/
│
├── app/
│   ├── api/
│   │   └── main.py
│   │
│   ├── generation/
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── rag_chain.py
│   │
│   ├── ingestion/
│   │   ├── chunking.py
│   │   ├── file_utils.py
│   │   ├── ingest.py
│   │   └── loaders.py
│   │
│   ├── observability/
│   │   └── tracing.py
│   │
│   ├── retrieval/
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── weaviate_client.py
│   │
│   ├── config.py
│   └── logging_config.py
│
├── evaluation/
│   ├── dataset.json
│   ├── evaluate.py
│   ├── generation_metrics.py
│   ├── retrieval_metrics.py
│   └── retrieval_diagnostics.py
│
├── ui/
│   └── streamlit_app.py
│
├── data/
│   └── uploads/
│
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| API | FastAPI |
| UI | Streamlit |
| Vector Database | Weaviate |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI |
| PDF Processing | PyMuPDF |
| Validation | Pydantic |
| Containers | Docker / Docker Compose |
| Retrieval | Hybrid BM25 + Vector Search |
| Evaluation | Custom frozen RAG evaluation suite |

---

## Engineering Decisions

### Why hybrid retrieval?

Semantic search captures conceptual similarity while BM25 captures exact terminology. Hybrid retrieval combines both signals.

### Why not use hybrid score as the hallucination guard?

Weaviate relative-score hybrid fusion normalizes scores within each candidate set. High scores therefore do not necessarily mean that a query is truly answerable from the document.

The LLM relevance gate is used for the final answerability decision.

### Why is reranking disabled?

The frozen evaluation showed lower Recall@5, MRR, and answer correctness after reranking, while also introducing another LLM request and additional latency.

### Why page-range metadata filtering?

Filtering before retrieval reduces irrelevant candidate chunks and gives users explicit control over document scope.

### Why cache query embeddings?

Identical queries otherwise generate identical embeddings repeatedly while incurring unnecessary API latency and cost.

### Why reuse the Weaviate client?

Opening a new connection for every retrieval request adds unnecessary network overhead. A shared process-level connection significantly reduces warm-query retrieval latency.

---

## Security

Sensitive configuration is stored using environment variables.

The repository intentionally excludes:

```text
.env
virtual environments
local uploads
Python caches
IDE configuration
```

Never commit:

```text
OpenAI API keys
Weaviate API keys
other credentials
```

Use `.env.example` as the configuration template.

---

## Current Production Configuration

```text
Hybrid retrieval          ON
Document filtering        ON
Page-range filtering      ON
Minimum score filtering   ON
Query embedding cache     ON
Shared Weaviate client    ON
LLM relevance gate        ON
Grounded generation       ON
Source citations          ON
Tracing                   ON
Reranking                 OFF
```

---

## Future Improvements

Potential extensions include:

- multi-document querying
- user authentication
- conversation-aware retrieval
- asynchronous ingestion jobs
- streaming generation
- Redis-backed distributed caching
- OpenTelemetry integration
- production API deployment
- automated CI evaluation
- larger benchmark datasets
- richer document metadata
- multi-tenant Weaviate collections

---

## Repository

https://github.com/sampath7013/weaviate-rag-application

---

## Author

**Sampath Kumar Muthyalapati**

Applied AI / Generative AI Engineer

Areas of focus:

```text
Retrieval-Augmented Generation
Generative AI
LLM Applications
Vector Databases
Agentic AI
AI Engineering
```