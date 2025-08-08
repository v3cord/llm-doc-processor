# api.py

from fastapi import FastAPI
from pydantic import BaseModel
from main import process_query  # Import your core logic

# Initialize the FastAPI app
app = FastAPI(
    title="LLM Document Processing API",
    description="An API to query unstructured documents using Google Gemini.",
    version="1.0.0"
)

# Define the request body model
class QueryRequest(BaseModel):
    query: str

# Define a root endpoint for health checks
@app.get("/", tags=["Health Check"])
async def root():
    return {"status": "ok", "message": "Welcome to the LLM Document Processing API!"}

# Define the main processing endpoint
@app.post("/process", tags=["Processing"])
async def process_document_query(request: QueryRequest):
    """
    Accepts a natural language query and returns a structured JSON decision
    based on the indexed documents.
    """
    response_json = process_query(request.query)
    return response_json

# To run this API server, use the following command in your terminal:
# uvicorn api:app --reload