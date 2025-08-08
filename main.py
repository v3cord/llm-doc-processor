# main.py

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List, Literal

load_dotenv()

# --- Pydantic Models for Structured Output ---
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

# --- LLM and Prompts ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

HYDE_TEMPLATE = """
Please write a short, hypothetical passage from an insurance policy document that would perfectly answer the user's question.
Focus on extracting key terms and concepts related to the user's situation.
USER QUESTION: {question}
PASSAGE:
"""

QA_TEMPLATE = """
You are an expert insurance claims analyst. Your task is to evaluate a query against a set of insurance policy clauses.
Based *only* on the provided context (policy clauses), generate a JSON response with the decision and justification.
CRITICAL INSTRUCTIONS:
1.  Pay special attention to any time-related conditions in the query, such as 'policy duration' or age, and actively look for corresponding 'waiting period' or 'age limit' clauses in the context.
2.  Analyze the user's situation based on the query.
3.  Scrutinize the provided context clauses to find all relevant rules.
4.  Make a final decision: "Approved", "Rejected", or "Needs More Information".
5.  Justify the decision by referencing the specific clauses. For each clause used, state its source, text, and how it applies (PASS/FAIL).
6.  Return the final answer strictly in the JSON format required.

CONTEXT (Policy Clauses):
{context}

QUERY:
{query}

REQUIRED JSON FORMAT:
{format_instructions}
"""

def format_docs(docs):
    """Helper function to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# --- THE CORE LOGIC - Returns a runnable chain ---
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
