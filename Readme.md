# 🧠 Mini-RAG Pro: Advanced Retrieval-Augmented Generation System

A high-performance, citation-aware RAG backend powered by **Pinecone, Google Gemini, and Cohere Rerank** — with async processing, caching, reranking, and a beautiful **Gradio UI**.

---

## ✨ Features

* **Smart Document Ingestion** → Chunking with overlap, metadata enrichment, and fault-tolerant embedding.
* **Hybrid Retrieval Pipeline** → Vector search (Pinecone) + Semantic reranking (Cohere) for highly relevant results.
* **Citation-Aware Answers** → Answers include inline citations `[1], [2]` with source attribution and relevance scores.
* **Asynchronous & Optimized** → Async ingestion, retrieval, and answer generation with thread pooling, caching, and retries.
* **Beautiful Gradio UI** → Modern, responsive interface with ingestion, query, batch, and stats panels.
* **Performance Monitoring** → Track ingestion time, query latency, error rates, and reliability metrics.
* **Production-Ready** → Error handling, timeouts, fallbacks, and health checks built-in.
* **Modular Design** → Clean separation of concerns: ingestion, retrieval, reranking, answering, caching.
* **FastAPI Backend** → RESTful API with CORS, validation, and clean structure.
* **Powered by Gemini** → Uses `gemini-1.5-flash` for fast, factual, and concise responses.

---

## ⚙️ Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mini-rag-pro.git
cd mini-rag-pro
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> ✅ Requires **Python 3.9+**

### 3. Environment Variables Setup

Create a file named `.env` in the project root:

```ini
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX=your_index_name_here
GOOGLE_API_KEY=your_gemini_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

* Get your keys:

  * Pinecone → [https://app.pinecone.io](https://app.pinecone.io)
  * Google AI Studio → [https://aistudio.google.com](https://aistudio.google.com)
  * Cohere → [https://dashboard.cohere.com](https://dashboard.cohere.com)

> ⚠️ Your Pinecone index must use: `768 dimensions` + `cosine similarity`

---

## 📡 API Endpoints

### 🔹 Ingest Documents

`POST /ingest`

```json
{
  "text": "Machine learning is a subset of artificial intelligence...",
  "source": "wikipedia",
  "title": "Introduction to ML"
}
```

### 🔹 Query

`POST /query`

```json
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

### 🔹 Batch Query

`POST /batch-query`

```json
{
  "queries": ["What is AI?", "Explain neural networks"],
  "top_k": 5
}
```

### 🔹 Health Check

`GET /health`

### 🔹 Stats

`GET /stats`

### 🔹 Performance Metrics

`GET /performance`

---

## 📥 Ingest Example

```bash
curl -X POST http://localhost:8000/ingest \
-H "Content-Type: application/json" \
-d '{
  "text": "Artificial intelligence is transforming industries.",
  "source": "tech-blog",
  "title": "AI Trends 2025"
}'
```

✅ Expected Response:

```json
{
  "status": "success",
  "message": "Successfully ingested 1 chunks",
  "details": {
    "chunks_ingested": 1,
    "processing_time": 1.2,
    "total_words": 6,
    "chunk_stats": {
      "avg_chunk_size": 6,
      "chunk_size_range": "6-6",
      "overlap_percentage": "15%",
      "embedding_dimensions": 768
    },
    "status": "success",
    "source": "tech-blog",
    "title": "AI Trends 2025",
    "chunk_ids": [0]
  }
}
```

---

## 🔍 Query Example

```bash
curl -X POST http://localhost:8000/query \
-H "Content-Type: application/json" \
-d '{
  "query": "What is artificial intelligence?",
  "top_k": 5
}'
```

✅ Example Response:

```json
{
  "status": "success",
  "answer": "Artificial intelligence is the simulation of human intelligence processes by machines [1]. It includes learning, reasoning, and self-correction [2].",
  "citations": [
    {
      "citation_id": 1,
      "source": "tech-blog",
      "title": "AI Trends 2025",
      "relevance_score": 0.92
    }
  ],
  "sources": [
    {
      "text": "Artificial intelligence is the simulation of human intelligence processes by machines...",
      "source": "tech-blog",
      "relevance_score": 0.92
    }
  ],
  "metadata": {
    "processing_time": 1.5,
    "documents_used": 2,
    "average_relevance_score": 0.89
  }
}
```

---

## 🎨 Gradio UI

### Run Locally

```bash
python gradio_ui.py
```

➡️ Open [http://localhost:7860](http://localhost:7860)

**UI Features:**

* 📥 Ingest with title & source
* 🔍 Query with citation display
* 📦 Batch query support
* 📊 Real-time system stats
* 🌙 Dark mode with gradient design

---

## 🚀 Deploy on Hugging Face Spaces

### 📂 Repository Structure

```
mini-rag-pro/
├── app.py
├── main.py
├── gradio_ui.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 🔑 Create `app.py` (Entry Point)

```python
import os, time, subprocess, threading, requests

def run_fastapi():
    print("🚀 Starting FastAPI backend...")
    subprocess.run(["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])

fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()

# Wait for FastAPI to start
def wait_for_fastapi(url="http://127.0.0.1:8000/health", timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(url).status_code == 200:
                print("✅ FastAPI is LIVE!")
                return True
        except:
            pass
        time.sleep(2)
    raise RuntimeError("❌ FastAPI failed to start")

print("⏳ Waiting for FastAPI...")
wait_for_fastapi()

print("🎨 Starting Gradio frontend...")
from gradio_ui import demo
demo.launch(server_name="0.0.0.0", server_port=7860)
```

### 🔧 Update `gradio_ui.py`

* Set `BACKEND_URL = "http://127.0.0.1:8000"`
* Use `gr.HTML()` instead of `gr.JSON()` for ingestion output.

### 📤 Deploy

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://huggingface.co/spaces/your-username/mini-rag-pro
git push -u origin main
```

### 🔐 Add Secrets

Go to **Settings → Secrets** in Hugging Face Spaces and add:

* `PINECONE_API_KEY`
* `PINECONE_INDEX`
* `GOOGLE_API_KEY`
* `COHERE_API_KEY`

✅ Your app will be live at:

```
https://your-username-mini-rag-pro.hf.space
```

---

## 📊 Tech Stack

* **FastAPI** → Backend API
* **Pinecone** → Vector DB
* **Cohere Rerank** → Semantic reranking
* **Google Gemini** → Answer generation
* **Gradio** → Interactive UI

---
