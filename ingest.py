import os
import re
import math
import zlib
from pathlib import Path
from typing import List, Dict, Any
from config import config
from database import VectorDatabase

class EmbeddingGenerator:
    """Generates vector embeddings for text chunks using local models or deterministic offline vectorization."""
    
    def __init__(self, dim: int = None):
        self.dim = dim or config.embedding_dim
        self._transformer_model = None
        self._init_model()

    def _init_model(self):
        """Attempts to load sentence-transformers embedding model if available."""
        try:
            from sentence_transformers import SentenceTransformer
            self._transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dim = 384
        except Exception:
            self._transformer_model = None

    def _stable_hash(self, key: str) -> int:
        """Deterministic hash calculation across different process launches."""
        return zlib.crc32(key.encode('utf-8'))

    def embed_text(self, text: str) -> List[float]:
        """Generates a normalized embedding vector for input text."""
        if self._transformer_model is not None:
            vec = self._transformer_model.encode(text)
            return vec.tolist()
        
        vec = [0.0] * self.dim
        words = re.findall(r'\w+', text.lower())
        if not words:
            return vec
            
        for i, word in enumerate(words):
            h1 = self._stable_hash(word) % self.dim
            vec[h1] += 1.0
            if i > 0:
                h2 = self._stable_hash(f"{words[i-1]}_{word}") % self.dim
                vec[h2] += 1.5
                
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

def extract_text_from_pdf(file_path: Path) -> str:
    """Extracts plain text from PDF files."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n\n"
        return text
    except Exception:
        with open(file_path, "rb") as f:
            raw = f.read().decode("latin1", errors="ignore")
        txt_parts = re.findall(r'\((.*?)\)\s*TJ|\((.*?)\)\s*Tj', raw)
        extracted = []
        for p in txt_parts:
            s = p[0] if p[0] else p[1]
            if len(s) > 2 and not s.startswith('/'):
                extracted.append(s)
        return "\n".join(extracted) if extracted else ""

def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
    """Splits a document text into overlapping paragraph-level chunks."""
    chunk_size = chunk_size or config.chunk_size
    chunk_overlap = chunk_overlap or config.chunk_overlap
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{p}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(p) > chunk_size:
                words = p.split()
                sub_chunk = []
                sub_len = 0
                for w in words:
                    sub_chunk.append(w)
                    sub_len += len(w) + 1
                    if sub_len >= chunk_size:
                        chunks.append(" ".join(sub_chunk))
                        overlap_words = sub_chunk[-(chunk_overlap // 6):] if len(sub_chunk) > 5 else []
                        sub_chunk = list(overlap_words)
                        sub_len = sum(len(w) + 1 for w in sub_chunk)
                if sub_chunk:
                    current_chunk = " ".join(sub_chunk)
            else:
                current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks if chunks else [text]

def run_ingestion(docs_dir: Path = None, db: VectorDatabase = None) -> Dict[str, Any]:
    """Scans document directory, chunks documents, generates embeddings, and populates database."""
    docs_dir = docs_dir or config.docs_dir
    db = db or VectorDatabase()
    embedder = EmbeddingGenerator()
    
    db.clear_db()
    total_docs = 0
    total_chunks = 0
    
    for file_path in docs_dir.glob("*.*"):
        ext = file_path.suffix.lower()
        if ext in [".md", ".txt", ".pdf"]:
            try:
                if ext == ".pdf":
                    content = extract_text_from_pdf(file_path)
                else:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                
                if not content.strip():
                    continue
                    
                passages = chunk_text(content)
                filename = file_path.name
                
                batch = []
                for idx, passage in enumerate(passages):
                    embedding = embedder.embed_text(passage)
                    batch.append({
                        "filename": filename,
                        "chunk_index": idx,
                        "content": passage,
                        "embedding": embedding
                    })
                
                db.insert_batch(batch)
                total_docs += 1
                total_chunks += len(passages)
            except Exception as e:
                print(f"[Ingest Error] Failed to process {file_path.name}: {e}")

    stats = db.get_stats()
    return {
        "processed_documents": total_docs,
        "processed_chunks": total_chunks,
        "database_stats": stats
    }

if __name__ == "__main__":
    print("Starting document ingestion pipeline...")
    result = run_ingestion()
    print(f"Ingestion complete!")
    print(f"Documents processed: {result['processed_documents']}")
    print(f"Chunks indexed: {result['processed_chunks']}")
    print(f"Database location: {result['database_stats']['db_path']}")
