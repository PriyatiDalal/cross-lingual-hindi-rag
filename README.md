# Cross-Lingual Hindi RAG Engine

A production-grade Retrieval-Augmented Generation (RAG) system capable of ingesting raw Hindi documents, performing normalized chunking, producing cross-lingual embeddings, and accurately answering queries in both Hindi and English with precise citations.

## 1. Chunking Strategy
- **Chunk Size & Overlap:** Chunks are bounded to ~500 characters with an 80-character overlap.
- **Justification:** Devanagari text carries dense semantic information per sentence. Respecting sentence endings (Hindi danda `।` and standard periods `.`) prevents breaking entity references across boundaries, while overlap preserves continuity across sliding contexts.

## 2. Vector Database Choice
- **ChromaDB:** Chosen because it is lightweight, open-source, runs natively embedded in Python without external daemon setups, and supports cosine distance similarity metrics.

## 3. Hindi Text Handling & Cross-Lingual Retrieval
- **Unicode NFC Normalization:** Applied via `unicodedata.normalize('NFC', text)` to eliminate variant codepoint mismatches in Devanagari ligatures.
- **Cross-Lingual Embeddings:** Utilizes `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (or `BAAI/bge-m3`), mapping both Hindi and English tokens into a unified multilingual vector space.

## 4. Limitations & Future Improvements
- **Table Parsing:** Complex markdown and OCR-less PDF table parsing can be improved using layout-aware tools such as `pdfplumber` or `marker-pdf`.
- **Hybrid Search:** Adding BM25 keyword matching combined with dense vector retrieval (Reciprocal Rank Fusion) would further improve acronym resolution (e.g., SLV-III, IGMDP).

## 5. How to Run

1. **Clone repository:**
   ```bash
   git clone <YOUR_REPO_URL>
   cd <REPO_FOLDER>