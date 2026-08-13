import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "sample_documents"
DB_PATH = DATA_DIR / "vector_store.db"

# Ensure data directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class RAGConfig:
    db_path: Path = DB_PATH
    docs_dir: Path = DOCS_DIR
    
    # Chunking Configuration
    chunk_size: int = 400
    chunk_overlap: int = 60
    
    # Retrieval Configuration
    top_k: int = 3
    similarity_threshold: float = 0.15  # Score floor above hash collision noise (~0.08)
    
    # Model Configuration
    embedding_dim: int = 384
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")  # "foundry", "openai", "fallback", or "auto"
    foundry_model_name: str = "phi-3.5-mini"
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "ollama")
    
    # Prompt Templates
    system_prompt: str = (
        "You are a helpful, precise offline AI Knowledge Assistant powered by RAG and Microsoft Foundry Local concepts.\n"
        "Your task is to answer the user's question using ONLY the provided document context passages.\n"
        "Strict Guidelines:\n"
        "1. Answer based strictly on the provided Context. Do NOT use outside knowledge.\n"
        "2. If the context does not contain enough information to answer the question, clearly state:\n"
        "   'I do not have enough information in the local knowledge base to answer this question.'\n"
        "3. Always cite the document source (e.g. [Source: filename.md]) when stating facts from context.\n"
        "4. Keep your answer clear, accurate, and concise."
    )

config = RAGConfig()
