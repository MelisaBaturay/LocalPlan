# Local RAG Architecture Overview

## What is Retrieval-Augmented Generation (RAG)?
Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Models (LLMs) by retrieving relevant facts from an external document repository before generating answers.

## Three Pillars of RAG
1. **Retrieve**: When a user poses a question, the question is converted into a vector embedding and matched against a local document vector store to extract the top-K relevant text passages.
2. **Augment**: The retrieved text passages are injected into the system/user prompt as grounded context alongside strict role instructions.
3. **Generate**: The local LLM processes the augmented prompt and generates an answer strictly grounded in the retrieved documents.

## Benefits of Local RAG
- **Zero Hallucination Risk**: Answers are bounded by local source text rather than pre-trained web assumptions.
- **Source Citation**: Answers can explicitly credit the document filename and paragraph section.
- **Offline & Zero Cloud Cost**: Uses local SQLite vector stores and local model runtimes, costing $0 in cloud API tokens.

## SQLite for Vector Storage
SQLite is a serverless, single-file relational database engine widely deployed across operating systems. When augmented with cosine similarity calculation routines in Python, SQLite serves as a lightweight, reliable local vector database for storing text chunks and array embeddings.
