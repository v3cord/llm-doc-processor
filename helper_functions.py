# helper_functions.py

import io
import requests
# We will use pypdf directly
from pypdf import PdfReader
from langchain.docstore.document import Document
# The rest of the imports are the same
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def load_document_from_url(url: str):
    """
    Downloads a PDF from a URL, reads it from memory using pypdf,
    and returns it as a list of LangChain Document objects.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Use io.BytesIO to treat the downloaded binary content as an in-memory file
        pdf_file = io.BytesIO(response.content)

        # Use PdfReader from pypdf to read the in-memory file
        reader = PdfReader(pdf_file)

        documents = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:  # Ensure the page has text content
                # Manually create a LangChain Document object for each page
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source_url": url, "page": i}
                    )
                )

        print(f"✅ Document loaded successfully from URL. Number of pages: {len(documents)}")
        return documents
    except Exception as e:
        print(f"Error loading document from URL: {e}")
        raise

# NO CHANGES ARE NEEDED FOR THE FUNCTIONS BELOW THIS LINE
def create_retriever_for_document(documents):
    """Takes a list of LangChain documents, creates an in-memory FAISS index,
    and returns a retriever object."""
    if not documents:
        print("⚠️ No documents to process. Returning None.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"✅ Document split into {len(texts)} chunks.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    faiss_index = FAISS.from_documents(texts, embeddings)

    retriever = faiss_index.as_retriever(search_kwargs={"k": 5})
    return retriever

def get_answer_for_question(question: str, retriever):
    """Runs a RAG chain for a single question against a given retriever."""
    if retriever is None:
        return "Could not process the document to create a retriever."

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
