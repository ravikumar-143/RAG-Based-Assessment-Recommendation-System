# 🚀 RAG-Based Assessment Recommendation System

A production-ready Retrieval-Augmented Generation (RAG) system that recommends the most relevant SHL assessments based on job roles and skills using **FAISS**, **Sentence Transformers**, **Ollama LLM**, **FastAPI**, and **Streamlit**.

---

# 🔍 Overview

This project helps recruiters and hiring managers quickly identify the most suitable SHL assessments for a given job role or skill set.

The system:

- Scrapes the SHL assessment catalog
- Builds vector embeddings using Sentence Transformers
- Stores embeddings in a FAISS vector database
- Retrieves the most relevant assessments using semantic search
- Reranks results using a local Ollama LLM
- Removes report-only assessments
- Exposes a FastAPI backend
- Provides an interactive Streamlit frontend

---

# ✨ Features

- Semantic Search using FAISS
- Local LLM Reranking using Ollama (Qwen 2.5)
- FastAPI REST API
- Streamlit User Interface
- Duplicate Removal
- Report Filtering
- Query Expansion for Job Roles
- Modular Project Structure
- Fully Local Deployment (No API Key Required)

---

# 🛠 Tech Stack

- Python 3.11+
- FastAPI
- Streamlit
- FAISS
- Sentence Transformers
- Ollama
- Qwen2.5
- Pandas
- BeautifulSoup
- Requests
- NumPy

---

# 📂 Project Structure

```
RAG-Based-Assessment-Recommendation-System/

│
├── api/
│   └── app.py
│
├── frontend/
│   └── app.py
│
├── retrieval/
│   ├── search.py
│   └── pipeline.py
│
├── reranker/
│   ├── rerank.py
│   └── domain_detection.py
│
├── embeddings/
│   ├── embedder.py
│   └── index_builder.py
│
├── scraper/
│   └── shl_scraper.py
│
├── evaluation/
│   └── evaluate.py
│
├── submission/
│   └── generate_submission.py
│
├── data/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/ravikumar-143/RAG-Based-Assessment-Recommendation-System.git
```

Move into the project

```bash
cd RAG-Based-Assessment-Recommendation-System
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🤖 Install Ollama

Download Ollama

https://ollama.com/download

Pull the model

```bash
ollama pull qwen2.5:1.5b
```

Verify

```bash
ollama list
```

---

# 📦 Build Embeddings

```bash
python embeddings/index_builder.py
```

---

# 🚀 Run FastAPI

```bash
python -m uvicorn api.app:app --port 8001
```

API Documentation

```
http://127.0.0.1:8001/docs
```

---

# 🎨 Run Streamlit

```bash
streamlit run frontend/app.py
```

Open

```
http://localhost:8501
```

---

# 🔌 API

## POST /recommend

### Request

```json
{
    "query": "Python Developer with SQL"
}
```

### Response

```json
{
    "recommended_assessments": [
        {
            "name": "Python (New)",
            "url": "...",
            "description": "...",
            "duration": null,
            "adaptive_support": "No",
            "remote_support": "No",
            "test_type": ["K"]
        }
    ]
}
```

---

# 🧠 Retrieval Pipeline

```
User Query
      │
      ▼
Query Expansion
      │
      ▼
Sentence Transformer
      │
      ▼
FAISS Vector Search
      │
      ▼
Top 50 Results
      │
      ▼
Report Filtering
      │
      ▼
Ollama LLM Reranking
      │
      ▼
Top 10 Assessments
      │
      ▼
FastAPI
      │
      ▼
Streamlit UI
```

---

# 📸 Application Screenshots

## Home Page

(Add Screenshot Here)

---

## Recommendations

(Add Screenshot Here)

---

## FastAPI Swagger

(Add Screenshot Here)

---

# 📊 Features Implemented

- SHL Assessment Scraper
- Semantic Search
- FAISS Vector Database
- Sentence Transformer Embeddings
- Query Expansion
- Report Filtering
- Duplicate Removal
- LLM Reranking
- FastAPI Backend
- Streamlit Frontend
- REST API
- Local LLM Deployment
- Evaluation Scripts

---

# 📈 Future Improvements

- PDF Job Description Upload
- Resume Parsing
- Hybrid Search (BM25 + FAISS)
- Docker Deployment
- Azure Deployment
- Kubernetes Support
- User Authentication
- Caching
- Feedback Learning

---

# 👨‍💻 Author

**K. Ravi Kumar**

📧 Email: kravik640@gmail.com

🔗 LinkedIn:
https://www.linkedin.com/in/kosgi-ravi-kumar-2a009a282

🔗 GitHub:
https://github.com/ravikumar-143

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
