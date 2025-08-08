# app.py - The Only Code File You Need

import streamlit as st
import os
import re
import asyncio
from dotenv import load_dotenv
from typing import List, Literal

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# --- Load API Key ---
load_dotenv()
if os.getenv("GOOGLE_API_KEY") is None:
    st.error("GOOGLE_API_KEY is not set. Please add it to your secrets.", icon="🚨")
    st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Query System",
    page_icon="🔎",
    layout="wide"
)
st.title("🔎 AI Document Query System")

# --- Pydantic Models for Structured Output ---
class Rule(BaseModel):
    clause_id: str = Field(description="e.g., 'Clause 12.3' or 'N/A'")
    source_document: str = Field(description="The source filename of the document")
    page: int = Field(description="The page number of the clause")
    text: str = Field(description="The exact text of the clause")
    evaluation: Literal['PASS', 'FAIL'] = Field(description="PASS or FAIL based on the query")
    reason: str = Field(description="Explanation of the evaluation")

class Justification(BaseModel):
    summary: str = Field(description="A brief summary of the final decision")
    rules_applied: List[Rule] = Field(description="A list of all rules applied")

class FinalResponse(BaseModel):
    decision: Literal['Approved', 'Rejected', 'Needs More Information']
    amount: int = Field(description="The approved amount, or 0")
    justification: Justification

# --- Helper & Logic Functions ---
def load_and_chunk_document(file_path: str):
    """Loads and chunks a single document."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = UnstructuredWordDocumentLoader(file_path)
    else:
        return None
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

@st.cache_resource
def load_or_create_retriever(_selected_filename):
    """Creates and caches a FAISS retriever for the selected file."""
    DATA_DIR = "data"
    DB_DIR_BASE = "db_faiss_cache"
    os.makedirs(DB_DIR_BASE, exist_ok=True)
    
    sanitized_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', _selected_filename)
    db_path = os.path.join(DB_DIR_BASE, sanitized_name)
    file_path = os.path.join(DATA_DIR, _selected_filename)
    
    # Fix for asyncio event loop error in Streamlit's threading context
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    if os.path.exists(db_path):
        faiss_index = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        with st.spinner(f"First time seeing '{_selected_filename}'. Creating knowledge base..."):
            texts = load_and_chunk_document(file_path)
            if texts:
                faiss_index = FAISS.from_documents(texts, embeddings)
                faiss_index.save_local(db_path)
            else:
                st.error("Could not process the document.")
                st.stop()
    
    return faiss_index.as_retriever(search_kwargs={"k": 5})

# --- Main Application UI ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
available_docs = [f for f in os.listdir(DATA_DIR) if f.endswith((".pdf", ".docx"))]

if not available_docs:
    st.warning(f"No documents found in the '{DATA_DIR}' folder. Please add your PDF or DOCX files.")
else:
    selected_doc = st.sidebar.selectbox("Choose a document to query:", available_docs)
    
    if selected_doc:
        retriever = load_or_create_retriever(selected_doc)
        st.sidebar.success(f"Knowledge base for '{selected_doc}' is ready.")
        
        # Define LLM and Prompts here as they are needed for the chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
        
        QA_TEMPLATE = """
        You are an expert insurance claims analyst... (rest of the detailed QA prompt)
        CONTEXT (Policy Clauses): {context}
        QUERY: {query}
        REQUIRED JSON FORMAT: {format_instructions}
        """

        parser = JsonOutputParser(pydantic_object=FinalResponse)
        qa_prompt = PromptTemplate(
            template=QA_TEMPLATE,
            input_variables=["context", "query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "query": RunnablePassthrough()}
            | qa_prompt
            | llm
            | parser
        )
        
        query = st.text_area("Enter your query:", placeholder="e.g., A 46-year-old male needs knee surgery...")

        if st.button("Process Query"):
            if query:
                with st.spinner("Analyzing..."):
                    try:
                        result = rag_chain.invoke(query)
                        st.json(result)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please enter a query.")
