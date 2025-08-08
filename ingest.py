# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Import FAISS instead of Chroma
from langchain_community.vectorstores import FAISS

def load_and_chunk_document(file_path: str):
    """Loads a single local document and splits it into chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_and_save_faiss_index(texts, db_path: str):
    """Creates and saves a FAISS vector index for the given texts."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create the FAISS index from the documents and embeddings
    faiss_index = FAISS.from_documents(texts, embeddings)
    # Save the index locally to the specified path
    faiss_index.save_local(db_path)
    return faiss_index
