import os
import re
import unicodedata
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)


def normalize_hindi_text(text: str) -> str:
    """Normalize Devanagari Unicode to NFC and clean extra whitespaces."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_and_chunk_pdf(pdf_path: str, chunk_size: int = 500, overlap: int = 80):
    reader = PdfReader(pdf_path)
    chunks = []
    chunk_counter = 1

    current_section = "परिचय (Introduction)"
    section_pattern = re.compile(r"^\s*(\d+[\s\.\•\-\—]+[^\n]+)")

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        normalized_text = normalize_hindi_text(raw_text)

        lines = normalized_text.split("\n")
        page_body = []
        for line in lines:
            line_clean = line.strip()
            match = section_pattern.match(line_clean)
            if match and len(line_clean) < 80:
                current_section = match.group(1).strip()
            else:
                page_body.append(line_clean)

        full_page_text = " ".join(page_body)

        sentences = re.split(r"(?<=[।\.])\s+", full_page_text)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk.strip():
                    chunks.append({
                        "chunk_id": f"chunk_{chunk_counter}",
                        "text": current_chunk.strip(),
                        "page": page_num,
                        "section": current_section,
                    })
                    chunk_counter += 1
                    
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = (overlap_text + " " + sentence).strip()

        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "text": current_chunk.strip(),
                "page": page_num,
                "section": current_section,
            })
            chunk_counter += 1

    return chunks


class HindiRAGSystem:
    def __init__(self, collection_name="hindi_kalam_rag"):
        
        self.embed_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.chroma_client = chromadb.Client()
        try:
            self.chroma_client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def index_document(self, pdf_path: str):
        chunks = extract_and_chunk_pdf(pdf_path)
        texts = [c["text"] for c in chunks]
        embeddings = self.embed_model.encode(texts, show_progress_bar=False).tolist()
        ids = [c["chunk_id"] for c in chunks]
        metadatas = [{"page": c["page"], "section": c["section"], "chunk_id": c["chunk_id"]} for c in chunks]

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Successfully indexed {len(chunks)} chunks from {pdf_path}.")

    def retrieve(self, query: str, top_k: int = 3):
        query_norm = normalize_hindi_text(query)
        query_vec = self.embed_model.encode([query_norm]).tolist()
        results = self.collection.query(query_embeddings=query_vec, n_results=top_k)

        retrieved_docs = []
        for i in range(len(results["ids"][0])):
            doc_id = results["ids"][0][i]
            text = results["documents"][0][i]
            meta = results["metadatas"][0][i]
            
            distance = results["distances"][0][i] if "distances" in results else 0.0
            score = round(1.0 - distance, 2)
            retrieved_docs.append({
                "chunk_id": doc_id,
                "text": text,
                "page": meta["page"],
                "section": meta["section"],
                "score": score
            })
        return retrieved_docs

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        context_str = "\n\n".join([
            f"[Chunk ID: {d['chunk_id']} | Page: {d['page']} | Section: {d['section']}]\n{d['text']}"
            for d in retrieved_docs
        ])

        prompt = f"""You are a precise factual assistant. Answer the user query based ONLY on the provided context below.
If the answer is not contained in the context, state 'जानकारी दस्तावेज़ में उपलब्ध नहीं है।'

Context:
{context_str}

User Query: {query}

Instructions:
- Provide a clear, direct, and concise answer.
- Answer in the same language as the query (Hindi for Hindi queries, English for English queries).
- Do not add outside knowledge.

Answer:"""

        try:
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception:
            # Fallback direct snippet extraction if API key is not configured
            return retrieved_docs[0]["text"]
