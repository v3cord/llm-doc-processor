# query_basic.py

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

DB_PATH = "db"

def test_basic_query():
    """
    Tests the basic retrieval and question-answering functionality.
    """
    # Initialize embeddings and load the vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    # Initialize the retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Initialize a modern and reliable Google Gemini Chat LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

    # Create the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # --- Ask a question ---
    query = "What is the waiting period for knee surgery?"
    print(f"Querying the system with: '{query}'")
    
    result = qa_chain.invoke({"query": query})

    # Print the results
    print("\n--- Answer ---")
    print(result['result'])
    print("\n--- Source Documents ---")
    for doc in result['source_documents']:
        print(f"Source: {doc.metadata['source']}, Page: {doc.metadata['page']}")
        print("-" * 20)


if __name__ == "__main__":
    test_basic_query()