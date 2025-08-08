# main.py

import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()

# --- Pydantic Models for Structured Output (No changes here) ---
class Rule(BaseModel):
    clause_id: str = Field(description="e.g., 'Clause 12.3' or 'N/A' if not specified in the text")
    source_document: str = Field(description="The source filename of the document")
    page: int = Field(description="The page number of the clause")
    text: str = Field(description="The exact text of the clause")
    evaluation: Literal['PASS', 'FAIL'] = Field(description="PASS or FAIL based on the query")
    reason: str = Field(description="Explanation of how this clause led to the evaluation for the given query")

class Justification(BaseModel):
    summary: str = Field(description="A brief summary of the final decision")
    rules_applied: List[Rule] = Field(description="A list of all rules that were applied to reach the decision")

class FinalResponse(BaseModel):
    decision: Literal['Approved', 'Rejected', 'Needs More Information'] = Field(description="The final decision")
    amount: int = Field(description="The approved amount, or 0 if rejected/needs more info")
    justification: Justification

# --- LLM and Prompts (No changes here) ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

HYDE_TEMPLATE = "..." # The HyDE prompt remains the same
QA_TEMPLATE = "..."   # The QA prompt remains the same

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- THE CORE LOGIC - NOW ACCEPTS A RETRIEVER ---
def get_processing_chain(retriever):
    """
    Creates the full processing chain, now dependent on the provided retriever.
    """
    parser = JsonOutputParser(pydantic_object=FinalResponse)
    
    hyde_prompt = PromptTemplate(input_variables=["question"], template=HYDE_TEMPLATE)
    qa_prompt = PromptTemplate(
        template=QA_TEMPLATE,
        input_variables=["context", "query"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    retrieval_chain = (
        hyde_prompt
        | llm
        | StrOutputParser()
        | retriever
        | format_docs
    )

    main_chain = (
        {
            "context": retrieval_chain,
            "query": RunnablePassthrough()
        }
        | qa_prompt
        | llm
        | parser
    )
    
    return main_chain