import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from config import config
from database import VectorDatabase
from retrieval import Retriever
from llm_provider import BaseLLMProvider, get_llm_provider

@dataclass
class RAGResponse:
    question: str
    answer: str
    retrieved_chunks: List[Dict[str, Any]]
    llm_provider: str
    latency_seconds: float
    has_sufficient_context: bool

class RAGEngine:
    """Core RAG pipeline orchestrator connecting retrieval, prompt synthesis, and LLM inference."""
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        system_prompt: Optional[str] = None
    ):
        self.retriever = retriever or Retriever()
        self.llm_provider = llm_provider or get_llm_provider()
        self.system_prompt = system_prompt or config.system_prompt

    def set_llm_provider(self, provider: BaseLLMProvider):
        """Allows dynamically switching the LLM provider at runtime."""
        self.llm_provider = provider

    def ask(
        self,
        question: str,
        top_k: int = None,
        similarity_threshold: float = None
    ) -> RAGResponse:
        """Executes the full RAG pipeline for a given user question."""
        start_time = time.time()
        top_k = top_k or config.top_k
        similarity_threshold = similarity_threshold or config.similarity_threshold
        
        # 1. Retrieve grounded context chunks
        chunks = self.retriever.get_top_chunks(
            query=question,
            top_k=top_k,
            threshold=similarity_threshold
        )
        
        has_sufficient_context = len(chunks) > 0
        
        # 2. Format context string
        if has_sufficient_context:
            context_blocks = []
            for chunk in chunks:
                match_pct = round(chunk['score'] * 100, 1)
                block = f"[Source Document: {chunk['filename']} | Section: Chunk #{chunk['chunk_index']} | Similarity Match: {match_pct}%]\n{chunk['content']}"
                context_blocks.append(block)
            formatted_context = "\n\n---\n\n".join(context_blocks)
        else:
            formatted_context = "No relevant document passages found in the local database."

        # 3. Construct user prompt for LLM
        user_prompt = (
            f"Context Passages:\n{formatted_context}\n\n"
            f"Question: {question}\n\n"
            f"Instructions:\n"
            f"- Answer the question using ONLY the provided context passages above.\n"
            f"- Include source document citations [Source: filename.md] for your answers.\n"
            f"- If the context does not contain the answer, say 'I do not have enough information in the local knowledge base to answer this question.'"
        )

        # 4. Generate LLM answer
        try:
            answer = self.llm_provider.generate_response(
                system_prompt=self.system_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            answer = f"Error generating answer: {e}"

        latency = round(time.time() - start_time, 3)

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            llm_provider=self.llm_provider.provider_name,
            latency_seconds=latency,
            has_sufficient_context=has_sufficient_context
        )

if __name__ == "__main__":
    from ingest import run_ingestion
    run_ingestion()
    
    engine = RAGEngine()
    response = engine.ask("What is Microsoft Foundry Local?")
    print(f"=== Question ===")
    print(response.question)
    print(f"\n=== Answer ({response.llm_provider}) [{response.latency_seconds}s] ===")
    print(response.answer)
