# Python Code Quality & Project Structure

## Modular Project Design
When building modular AI applications in Python, keep components separated into distinct responsibilities:
- `config.py`: Global constants, prompt templates, paths, and environment settings.
- `database.py`: Database connection lifecycle, table schema initialization, and SQL execution.
- `ingest.py`: Document ingestion, chunking, and embedding generation.
- `retrieval.py`: Vector distance calculation and top-K passage selection.
- `llm_provider.py`: Clean interface isolating LLM API calls from application business logic.
- `rag_engine.py`: Higher-level workflow connecting retrieval and LLM generation.

## Code Style & Type Annotations
- Use Python standard `dataclasses` or `pydantic` models for structured data transfer.
- Always include type hints (e.g. `def retrieve(query: str, top_k: int = 3) -> List[Dict[str, Any]]:`).
- Avoid global mutable state. Instantiate database handlers or configuration objects explicitly.

## Exception Handling & Logging
- Gracefully handle missing files and corrupted database instances.
- Provide fallbacks when hardware acceleration or specific model packages are absent on student machines.
