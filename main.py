# main.py
import os
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, HttpUrl
from typing import List
from dotenv import load_dotenv
from helper_functions import load_document_from_url, create_retriever_for_document, get_answer_for_question

load_dotenv()

if os.getenv("GOOGLE_API_KEY") is None:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

class HackRxRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

app = FastAPI(
    title="HackRx 6.0 Submission API",
    description="Processes documents and questions for the hackathon.",
)

@app.post("/hackrx/run", response_model=HackRxResponse)
async def process_hackrx_request(request: HackRxRequest, authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        print("Warning: Authorization header missing or malformed.")

    try:
        docs = load_document_from_url(str(request.documents))
        retriever = create_retriever_for_document(docs)
        
        answers = []
        for question in request.questions:
            answer = get_answer_for_question(question, retriever)
            answers.append(answer)
        
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
