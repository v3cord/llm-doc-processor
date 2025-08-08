# helper_functions.py

import io
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_document_from_url(url: str):
    """
    Downloads a PDF from a URL, loads it into memory, and returns it as a list
    of LangChain Document objects.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # Use io.BytesIO to treat the downloaded binary content as an in-memory file
        pdf_file = io.BytesIO(response.content)
        
        loader = PyPDFLoader(pdf_file)
        documents = loader.load()
        return documents
    except requests.exceptions.RequestException as e:
        print(f"Error downloading or reading the document from URL: {e}")
        raise

def create_retriever_for_document(documents):
    """
    Takes a list of LangChain documents, creates an in-memory vector store,
    and returns a retriever object.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Initialize the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create an in-memory vector store using Chroma. No need to persist for this use case.
    vector_store = Chroma.from_documents(texts, embeddings)
    
    # Create a retriever that finds the top 5 most relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

def get_answer_for_question(question: str, retriever):
    """
    Runs a Retrieval-Augmented Generation (RAG) chain for a single question
    against a given retriever and returns a concise string answer.
    """
    
    # This prompt template is crucial for forcing the LLM to answer only from the given text
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
    
    # Use a fast and capable model for question answering
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)
    
    # The RAG chain definition using LangChain Expression Language (LCEL)
    rag_chain = (
        # The retriever is invoked here with the question to get relevant documents
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser() # Extracts the string content from the LLM's response
    )
    
    # Invoke the chain to get the answer
    return rag_chain.invoke(question)
