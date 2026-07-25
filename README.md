# 🚀Assessment Recommendation Engine

A production-grade Retrieval-Augmented Generation (RAG) system that scrapes SHL Individual Test Solutions, builds vector embeddings with FAISS, reranks using Groq LLM, balances assessment domains, and serves recommendations via FastAPI and Streamlit.

---

## 🔍 Overview

This system:

- Scrapes SHL product catalog (type=1)
- Builds MiniLM embeddings + FAISS index
- Performs vector retrieval
- Reranks using Groq LLaMA (llama3-70b-8192)
- Applies domain balancing (tech / behavioral / cognitive)
- Exposes FastAPI endpoints
- Provides Streamlit frontend
- Generates submission CSV for evaluation

---

## ✅ Manager Checklist

- ✔ Catalog scraping (≥ 377 items validation)
- ✔ FAISS embedding index + metadata store
- ✔ RAG pipeline with Groq reranking
- ✔ Domain balancing logic
- ✔ Recall@10 evaluation
- ✔ FastAPI backend
- ✔ Streamlit frontend
- ✔ Submission CSV generator

---

## 🌐 Live Endpoints

> Update after deployment

- **Web App URL:** `<insert Streamlit public URL>`
- **API Endpoint:**
  
  ```http
  POST https://<your-api-host>/recommend
  ```

  **Request Body**
  ```json
  {
    "query": "text"
  }
  ```

---

## ⚙️ Quickstart Guide

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Set Environment Variables

Create `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 3️⃣ Scrape SHL Catalog (Validates ≥ 377 items)

```bash
python scraper/shl_scraper.py
```

### 4️⃣ Build Embeddings + FAISS Index

```bash
python embeddings/index_builder.py
```

### 5️⃣ (Optional) Run Evaluation

Populate:

```
data/train_dataset.csv
```

Then:

```bash
python evaluation/evaluate.py
```

### 6️⃣ Run FastAPI Backend

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 7️⃣ Run Streamlit Frontend

```bash
streamlit run frontend/app.py
```

If API is remote:

```
set API_URL=https://your-api-host
```

### 8️⃣ Generate Final Submission

Populate:

```
data/test_dataset.csv
```

Then:

```bash
python submission/generate_submission.py
```

Output:

```
submission/submission.csv
```

---

## 📂 Project Structure

```
scraper/
 └── shl_scraper.py

embeddings/
 └── index_builder.py

retrieval/
 ├── search.py
 └── pipeline.py

reranker/
 ├── domain_detection.py
 └── rerank.py

evaluation/
 └── evaluate.py

api/
 └── app.py

frontend/
 └── app.py

submission/
 └── generate_submission.py

scripts/
 └── sanity_check.py
```

---

## 📊 Data Artifacts

| File | Description |
|------|-------------|
| `data/cleaned_catalog.json` | Scraped SHL catalog (≥377 items) |
| `data/train_dataset.csv` | Training queries with ground truth |
| `data/test_dataset.csv` | Test queries |
| `embeddings/faiss.index` | FAISS vector index |
| `embeddings/metadata.json` | Metadata store |
| `submission/submission.csv` | Final output |

---

## 🔌 API Contract

### POST `/recommend`

**Input**

```json
{
  "query": "string"
}
```

**Output**

```json
{
  "recommended_assessments": [
    {
      "url": "",
      "name": "",
      "adaptive_support": "Yes/No",
      "description": "",
      "duration": null,
      "remote_support": "Yes/No",
      "test_type": ["..."]
    }
  ]
}
```

### Behavior

- Returns 1–10 results
- 400 → Empty query
- 404 → No results
- 500 → Index missing / internal error

---

## 🧠 Architecture

1. Scraper → Extract SHL catalog
2. Embeddings → MiniLM vector encoding
3. Retrieval → FAISS similarity search
4. Reranking → Groq LLaMA JSON-only contract
5. Domain Balancing → Mixed category coverage
6. API → FastAPI serving layer
7. UI → Streamlit client

---

## 🧪 Sanity Check

After scraping and building index:

```bash
python scripts/sanity_check.py
```

Validates:

- Catalog size
- Index presence
- Sample retrieval
- No LLM dependency

---

## 🛡 Notes

- Uses `lxml`, `BeautifulSoup`, `tenacity`
- Groq LLaMA 3 (70B) reranking with temperature=0
- Falls back to vector retrieval on LLM/API failure
- Streamlit displays:
  - Clickable URLs
  - Description
  - Duration
  - Adaptive & remote support
  - Test type tags

---

## 📈 Evaluation Metric

- Recall@10
- Baseline vs Rerank vs Balanced

---

## 📌 Submission Format

```
Query | Assessment_url
```

Generated via:

```bash
python submission/generate_submission.py
```

---

## 👨‍💻 Author

Teja Nadella  
GitHub: https://github.com/TejaNadella28

---

## ⭐ If You Found This Useful

Give it a star ⭐
