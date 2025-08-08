# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

def load_and_chunk_document(file_path: str):
    """Loads a single local document and splits it into chunks."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        # Make sure you have run: pip install "unstructured[docx]"
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        raise ValueError("Unsupported file type")
        
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    return texts

def create_and_persist_db(texts, db_path: str):
    """Creates and persists a Chroma vector database for the given texts."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # This creates the database and saves it to the db_path automatically
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=db_path
    )
    return vectordb
