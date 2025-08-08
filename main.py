# main.py
import os
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl
from typing import List
from dotenv import load_dotenv
from helper_functions import load_document_from_url, create_retriever_for_document, get_answer_for_question

# Load environment variables
load_dotenv()

# Check for API Key at startup
if os.getenv("GOOGLE_API_KEY") is None:
    # This will cause the app to fail on Render if the env var is not set, which is good.
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

# --- Pydantic Models for Request and Response ---
class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# --- FastAPI App Instance ---
# This is the line the error is about. It MUST be named 'app'.
app = FastAPI(
    title="HackRx 6.0 Submission API",
    description="Processes documents and questions for the hackathon.",
)

@app.post("/hackrx/run", response_model=HackRxResponse)
async def process_hackrx_request(request: HackRxRequest, authorization: str = Header(None)):
    """
    This endpoint receives a document URL and a list of questions,
    and returns a list of answers.
    """
    if not authorization or not authorization.startswith("Bearer "):
        print("Warning: Authorization header missing or malformed.")

    try:
        print(f"Loading document from: {request.documents}")
        docs = load_document_from_url(str(request.documents))

        print("Creating retriever for the document...")
        retriever = create_retriever_for_document(docs)

        answers = []
        for i, question in enumerate(request.questions):
            print(f"Processing question {i+1}/{len(request.questions)}: {question}")
            answer = get_answer_for_question(question, retriever)
            answers.append(answer)

        print("All questions processed successfully.")
        return HackRxResponse(answers=answers)

    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal error occurred: {e}"
        )

@app.get("/")
def read_root():
    return {"status": "API is running"}
