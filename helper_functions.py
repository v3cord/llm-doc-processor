# helper_functions.py

import io
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_document_from_url(url: str):
    """Downloads a PDF from a URL and loads it into memory."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_file = io.BytesIO(response.content)
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        return documents
    except requests.exceptions.RequestException as e:
        print(f"Error downloading or reading the document from URL: {e}")
        raise

def create_retriever_for_document(documents):
    """Takes a list of LangChain documents, creates an in-memory FAISS index,
    and returns a retriever object."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    faiss_index = FAISS.from_documents(texts, embeddings)
    
    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    return retriever

def get_answer_for_question(question: str, retriever):
    """Runs a RAG chain for a single question against a given retriever."""
    template = """
    Answer the following question based ONLY on the provided context.
    If the context does not contain the information to answer the question, state exactly: "Answer not found in the provided document."
    Provide a direct and concise answer. Do not add any extra information or introductory phrases.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain.invoke(question)
