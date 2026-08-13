import streamlit as st
import time
from config import config
from database import VectorDatabase
from ingest import run_ingestion
from rag_engine import RAGEngine
from llm_provider import get_llm_provider, FoundryLocalProvider, LocalOpenAIProvider, OfflineFallbackProvider

st.set_page_config(
    page_title="Local RAG AI Assistant - Microsoft Foundry Local",
    page_icon="⚡",
    layout="wide"
)

if "db" not in st.session_state:
    st.session_state.db = VectorDatabase()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your **Local RAG AI Assistant** running offline.", "chunks": []}
    ]

st.title("⚡ Local RAG Knowledge Assistant")
st.caption("Microsoft Foundry Local Architecture & SQLite Vector Storage")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask a question about your local documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        engine = RAGEngine()
        response = engine.ask(user_query)
        st.markdown(response.answer)

    st.session_state.messages.append({"role": "assistant", "content": response.answer, "chunks": response.retrieved_chunks})
