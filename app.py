# app.py

import streamlit as st
import os
import re
# Import the new FAISS function
from ingest import load_and_chunk_document, create_and_save_faiss_index
from main import get_processing_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Import FAISS instead of Chroma
from langchain_community.vectorstores import FAISS

st.set_page_config(
    page_title="Document Query System",
    page_icon="🔎",
    layout="wide"
)

st.title("🔎 Document Query System")
st.write("Select a document, ask a question, and get a structured answer based on its content.")

DATA_DIR = "data"
DB_DIR_BASE = "db_faiss_cache" # New cache directory for FAISS

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR_BASE, exist_ok=True)

def get_available_docs():
    """Scans the data directory for available documents."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith((".pdf", ".docx"))]

def sanitize_filename(filename):
    """Creates a safe directory name from a filename."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

@st.cache_resource
def load_or_create_retriever(_selected_filename):
    """
    Checks for a cached FAISS index for the selected file. If none exists, it creates one.
    Returns a LangChain retriever object.
    """
    sanitized_name = sanitize_filename(_selected_filename)
    db_path = os.path.join(DB_DIR_BASE, sanitized_name)
    file_path = os.path.join(DATA_DIR, _selected_filename)

    # Initialize embeddings once
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if os.path.exists(db_path):
        # Load from the local FAISS index
        faiss_index = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        # Create new index and save it
        texts = load_and_chunk_document(file_path)
        faiss_index = create_and_save_faiss_index(texts, db_path)

    return faiss_index.as_retriever(search_kwargs={"k": 5})

# --- Main Application UI ---
available_docs = get_available_docs()

if not available_docs:
    st.warning(f"No documents found in the '{DATA_DIR}' folder. Please add your PDF or DOCX files.")
else:
    st.sidebar.header("Select Document")
    selected_doc = st.sidebar.selectbox(
        "Choose a document to query:",
        available_docs
    )

    if selected_doc:
        st.header(f"Querying: `{selected_doc}`")

        with st.spinner(f"Loading knowledge base for '{selected_doc}'... This may take a moment on first load."):
            retriever = load_or_create_retriever(selected_doc)

        st.success(f"Knowledge base for '{selected_doc}' is ready to be queried.")

        processing_chain = get_processing_chain(retriever)

        query = st.text_area(
            "Enter your query here:",
            placeholder=f"e.g., What is the waiting period for knee surgery in {selected_doc}?",
            height=120
        )

        if st.button("Process Query"):
            if query:
                with st.spinner("Analyzing your query..."):
                    try:
                        result = processing_chain.invoke(query)
                        st.subheader("📋 Structured Response")
                        st.json(result)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a query.")
