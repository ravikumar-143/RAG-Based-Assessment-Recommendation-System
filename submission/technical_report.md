# SHL Assessment Recommendation Engine – Technical Report

## 1. Problem Statement
Build an automated recommender for SHL “Individual Test Solutions” that, given a role-related query, returns up to 10 relevant assessments. The system must scrape the SHL catalog, index content, retrieve via vector search, optionally rerank with an LLM, balance domains, and deliver results via API/UI and submission CSV.

## 2. Data Scraping Approach
- Target: SHL product catalog filtered to type=1 (Individual Test Solutions).
- Pagination: iterates `start` in steps of 12 with multi-pass retry on failures.
- Requests: session with retry/backoff, 8–10s timeout, light rate limit.
- Detail enrichment: fetch detail pages to collect meta description and duration; retry pass fills missing details.
- Output: `data/cleaned_catalog.json` + CSV (377 items, deduped by URL).

## 3. Embedding Strategy
- Model: `sentence-transformers/all-MiniLM-L6-v2` (768-dim, normalized).
- Text: `"{name} - {description}"` per item.
- Environment: `TRANSFORMERS_NO_TF=1` to avoid TF overhead.

## 4. Vector Retrieval Pipeline
- Index: FAISS `IndexFlatIP` over normalized embeddings (`embeddings/faiss.index`).
- Metadata: parallel `metadata.json` aligned to index order.
- Retrieval: encode query, cosine-sim search (inner product on normalized vectors), return top_k with scores.

## 5. Groq LLM Reranking Strategy
- Model: `llama3-70b-8192` via Groq API (requires `GROQ_API_KEY`).
- Prompt: System instructs to reorder given candidates only; no fabrication; return JSON array.
- Input fields: name, description, url, test_type, remote/adaptive flags.
- Fallback: if parsing fails, fall back to retrieval order.

## 6. Domain Balancing Logic
- Detect domains from query (technical, behavioral, cognitive, other) via keyword mapping.
- Round-robin selection across active domains to mix results; then fill remaining slots from ranked list while preserving metadata.
- Goal: ensure presence of knowledge/skills (K/ S) and personality/behavior (B/ P) where relevant.

## 7. Evaluation Results
- Metric: Recall@10 per query.
- Stages: baseline (vector), reranked (LLM), balanced (LLM + domain mix).
- Output format (per query): block showing all three recalls; summary means printed at end.
- Current note: training CSV now populated with 10 queries (multiple truths each); run `python evaluation/evaluate.py` to compute actual scores once Groq key is available.

## 8. Challenges Faced
- Unstable detail page fetches (timeouts, MissingSchema) → fixed with URL normalization, retries, and multi-pass pagination.
- Long model imports on Windows → mitigated via `TRANSFORMERS_NO_TF=1` and path setup.
- Need for full coverage (>=377 items) with descriptions → added detail backfill pass.

## 9. Future Improvements
- Add caching for Groq reranks to reduce latency/cost.
- Train a lightweight local reranker (cross-encoder) as fallback.
- Expand domain detection with learned classifier instead of keyword map.
- Add unit tests around validators and pipeline.
- Enhance duration parsing with structured page scraping instead of regex.
