# app.py

import streamlit as st
import os
import re
from ingest import load_and_chunk_document, create_and_persist_db
from main import get_processing_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Query System",
    page_icon="🔎",
    layout="wide"
)

# --- App Title and Description ---
st.title("🔎 Document Query System")
st.write("Select a document, ask a question, and get a structured answer based on its content.")

# --- Constants ---
DATA_DIR = "data"
DB_DIR_BASE = "db_cache"

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR_BASE, exist_ok=True)


def get_available_docs():
    """Scans the data directory for available documents."""
    return [f for f in os.listdir(DATA_DIR) if f.endswith((".pdf", ".docx"))]

def sanitize_filename(filename):
    """Creates a safe directory name from a filename."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

# --- Caching Logic ---
# This is now a "pure" data function with NO st. calls inside.
# It is safe to cache.
@st.cache_resource
def load_or_create_retriever(_selected_filename):
    """
    Checks for a cached DB for the selected file. If none exists, it creates one.
    Returns a LangChain retriever object.
    """
    sanitized_name = sanitize_filename(_selected_filename)
    db_path = os.path.join(DB_DIR_BASE, sanitized_name)
    file_path = os.path.join(DATA_DIR, _selected_filename)
    
    if os.path.exists(db_path):
        # Load from cache
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings)
    else:
        # Create new and cache
        texts = load_and_chunk_document(file_path)
        vectordb = create_and_persist_db(texts, db_path)
    
    return vectordb.as_retriever(search_kwargs={"k": 5})


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
        
        # UI elements like the spinner are now OUTSIDE the cached function.
        with st.spinner(f"Loading knowledge base for '{selected_doc}'... This may take a moment on first load."):
            retriever = load_or_create_retriever(selected_doc)
        
        st.success(f"Knowledge base for '{selected_doc}' is ready to be queried.")
        
        # Get the full processing chain
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