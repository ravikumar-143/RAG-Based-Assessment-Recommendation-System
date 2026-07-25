# Assessment Recommendation Engine

## 1. Introduction
A retrieval-augmented recommendation engine that matches SHL assessments to user queries using dense retrieval, optional LLM reranking, and domain-aware balancing. Built for reproducibility, transparency, and deployment readiness.

## 2. Problem Statement
Given a catalog of SHL assessments and user job/skill queries, return the top-N most relevant assessments while honoring product constraints (schema validity, domain diversity, and submission rules).

## 3. Data Scraping Strategy
- Scraped SHL product catalog pages (individual assessment solutions) into a cleaned catalog (~377 items).
- Stored structured fields (name, url, description, duration, test_type, remote/adaptive support) in `cleaned_catalog.json/csv`.
- Avoided modifying scraper per compliance requirement.

## 4. Embedding & Vector Index
- Model: `sentence-transformers/all-MiniLM-L6-v2` (frozen per requirement).
- Encoded catalog into dense vectors and built a FAISS IndexFlatIP (`embeddings/faiss.index`) with metadata mirror (`embeddings/metadata.json`).
- Index loads directly without rebuild for inference and deployment checks.

## 5. Retrieval Pipeline
- Baseline: FAISS similarity search over normalized MiniLM embeddings (top-k configurable).
- Pipeline entry: `retrieval.pipeline.recommend(query, top_k=30, max_results=10, strict_mode)`.
- Schema validation: every recommendation is validated to contain exactly the required keys before returning.

## 6. LLM Reranking Strategy
- Model: Groq `llama3-70b-8192`.
- Strict mode (used in evaluation/submission): raises on Groq failure; logs either "Groq reranking ACTIVE" or "Groq reranking FAILED".
- Fallback mode (non-strict) available for API robustness but still logs failures.

## 7. Domain Balancing Logic
- Detects domains (technical, behavioral, cognitive) from query keywords.
- Maps test_type codes to domains and performs round-robin selection.
- Enforces minimum 30% representation for the secondary domain on multi-domain queries to avoid dominance by a single domain.

## 8. Evaluation Results
- Metric: Recall@10 on training labels.
- Latest strict Groq evaluation (script writes `evaluation_results.txt`):
  - Baseline Mean Recall@10: **0.833**
  - Reranked Mean Recall@10: **0.833**
  - Balanced Mean Recall@10: **0.833**
- Per-query recalls are logged; balanced and reranked parity indicates retrieval already surfaces gold items at rank ≤10.

## 9. Observations
- Rerank did not increase mean recall because the baseline already hits relevant URLs within top-10; rerank preserved ordering without improving hit rate.
- Domain balancing provides diversity guarantees (not directly affecting Recall@10) and protects against single-domain over-selection for multi-domain intents.

## 10. Limitations
- Recall@10 is measured only on provided training labels; hidden or larger evaluation sets may differ.
- Groq reranking requires a valid `GROQ_API_KEY`; strict mode will fail fast if absent.
- Domain balancing relies on test_type metadata fidelity; incomplete tagging could reduce balancing quality.

## 11. Future Improvements
- Expand labeled data for broader coverage and better evaluation confidence.
- Experiment with cross-encoder rerankers to optimize beyond recall (e.g., NDCG, diversity).
- Add query understanding to auto-detect required assessment duration/constraints.
- Implement caching for Groq rerank results to reduce latency and cost.
- Extend monitoring/alerting around API schema validation and Groq health.
