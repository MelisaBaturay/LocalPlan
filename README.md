# Local RAG AI Assistant with Microsoft Foundry Local

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![SQLite](https://img.shields.io/badge/storage-SQLite-003B57.svg)](https://www.sqlite.org/)
[![Microsoft Foundry Local](https://img.shields.io/badge/AI-Microsoft%20Foundry%20Local-0078D4.svg)](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968)
[![Streamlit UI](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

A complete, production-grade, 100% offline **Local Retrieval-Augmented Generation (RAG) AI Assistant** built for computer science students and developers. Powered by **Microsoft Foundry Local** on-device inference, SQLite vector storage, and Streamlit, this project provides a zero-cloud Q&A chatbot that answers questions using local document collections without internet dependencies or cloud API costs.

---

## 📐 Project Architecture

```
                                +---------------------------+
                                |  Client Interface         |
                                |  - Streamlit Web Dashboard|
                                |  - Rich Interactive CLI   |
                                +-------------+-------------+
                                              |
                                              v
                                +---------------------------+
                                |     RAG Orchestrator      |
                                |   (rag_engine.py)         |
                                +------+--------------+-----+
                                       |              |
                    +------------------+              +------------------+
                    v                                                    v
     +------------------------------+                     +------------------------------+
     |     Retrieval Engine         |                     |        LLM Provider          |
     |   - Query Vectorization      |                     |  - Microsoft Foundry Local   |
     |   - Cosine Similarity Search |                     |  - Local OpenAI-Compatible   |
     |   - Top-K Chunk Filtering    |                     |  - Lightweight CPU Fallback  |
     +--------------+---------------+                     +------------------------------+
                    |
                    v
     +------------------------------+
     |   SQLite Vector Database     |
     |   - Document Chunks          |
     |   - Embeddings Storage       |
     |   - Metadata & Source Paths  |
     +------------------------------+
```

---

## 📁 Repository Structure

```
LocalPlan/
├── config.py                 # System parameters, database paths, and prompt templates
├── database.py               # SQLite vector database manager with cosine similarity
├── ingest.py                 # Document chunking pipeline & embedding vectorizer
├── retrieval.py              # Query vectorizer and top-K similarity search
├── llm_provider.py           # Multi-backend driver (Foundry Local, Ollama, Offline Fallback)
├── rag_engine.py             # RAG workflow orchestrator & prompt synthesizer
├── app.py                    # Streamlit Web Application UI (Dark glassmorphism theme)
├── cli.py                    # Interactive Rich Terminal Interface
├── test_rag.py               # Automated pytest suite
├── requirements.txt          # Python dependencies
└── data/
    └── sample_documents/     # Local knowledge base (.md, .txt)
        ├── cs101_course_faq.md
        ├── foundry_local_guide.md
        ├── rag_architecture_overview.md
        └── python_best_practices.md
```

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/your-username/LocalRAG-FoundryLocal.multi
cd LocalPlan
pip install -r requirements.txt
```

*(Optional) Install Microsoft Foundry Local SDK:*
```bash
pip install foundry-local-sdk
```

### 2. Ingest Document Knowledge Base
Populate the SQLite vector store with the provided sample documents:
```bash
python ingest.py
```

### 3. Launch Web UI (Streamlit)
Run the interactive web dashboard:
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` in your browser.

### 4. Launch Terminal CLI
Alternatively, chat with your local documents directly in the terminal:
```bash
python cli.py
```

### 5. Run Automated Tests
Run unit and integration tests to verify database, retrieval, and RAG logic:
```bash
pytest test_rag.py -v
```

---

## 📅 Student 4-Week / 6-Week Project Plan

This repository accompanies the **One-Month Local RAG Summer Program** designed for computer science students:

### Phase 1: Foundational Learning (Weeks 1–2)
- **Week 1: RAG Concepts & Foundry Local Setup**
  - Understand "Retrieve, Augment, Generate" workflow.
  - Install `foundry-local-sdk` and execute `python main.py` baseline completion.
  - Explore modular Python structure (`config.py`).
- **Week 2: Embeddings, Vector Search & SQLite**
  - Learn vector space models, cosine similarity distance.
  - Set up SQLite schema (`database.py`) storing array blobs.
  - Practice prompt engineering and system prompt rules.

### Phase 2: Project Implementation (Weeks 3–4)
- **Week 3: Ingestion & Retrieval Pipeline**
  - Implement overlapping chunking logic (`ingest.py`).
  - Code vector similarity search (`retrieval.py`).
  - Verify top-K query matching against sample documents.
- **Week 4: LLM Integration & UI Assembly**
  - Connect local model runtime (`llm_provider.py`).
  - Build Streamlit chat interface with live context inspector (`app.py`).
  - Enforce responsible AI guidelines and source citations (`rag_engine.py`).

### Phase 3: Testing, Evaluation & Presentation (Weeks 5–6)
- **Week 5: Testing & Benchmarking**
  - Run edge-case tests on answerable and unanswerable queries (`test_rag.py`).
  - Measure response latency and optimize top-K retrieval parameters.
- **Week 6: Final Presentation & Demo Day**
  - Finalize README and project documentation.
  - Present live demo showcasing zero-internet local Q&A capabilities.

---

## 📚 References & Resources

- [Microsoft Tech Community: Building Your First Local RAG Application with Foundry Local](https://techcommunity.microsoft.com/blog/azuredevcommunityblog/building-your-first-local-rag-application-with-foundry-local/4501968)
- [Microsoft Learn: Get started with Foundry Local](https://learn.microsoft.com/en-us/azure/)
- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
