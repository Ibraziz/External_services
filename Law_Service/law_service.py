from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from eurlex_client import EURLexClient
from contract_summary import analyze_legal_document, LegalAnalysis
from openai import OpenAI
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Law Service", description="RESTful API wrapper for EUR-Lex document search"
)

# Initialize EUR-Lex client (credentials loaded from .env)
try:
    eurlex_client = EURLexClient()
except ValueError as e:
    print(f"Warning: EUR-Lex client initialization failed: {e}")
    eurlex_client = None


# Request/Response Models
class Page(BaseModel):
    page_number: int
    text: str
    layout: Optional[list] = None


class Result(BaseModel):
    pages: Dict[str, Page]


class DocumentRequest(BaseModel):
    result: Result


class SearchRequest(BaseModel):
    query: str


class DocumentLink(BaseModel):
    type: str
    url: str


class Document(BaseModel):
    title: str
    language: str
    links: List[DocumentLink]
    reference: Optional[str] = None
    rank: Optional[int] = None


class SearchResponse(BaseModel):
    query: str
    success: bool
    total_hits: int
    num_results: int
    documents: List[Document]
    error: Optional[str] = None


class LegalAnalysisResponse(BaseModel):
    summary: str
    suggested_questions: List[str]
    search_responses: List[SearchResponse]


# Connect to local vLLM server
client = OpenAI(
    # api_key="dummy",
    # base_url="http://localhost:8080/v1"
    base_url="https://router.huggingface.co/v1",
    api_key="",
)


@app.post("/LegalAnalysis", response_model=LegalAnalysisResponse)
async def extract_law_keywords(request: DocumentRequest):
    """
    Analyze legal document and extract summary, keywords, and suggested questions.
    Uses the analyze_legal_document orchestrator from contract_summary module.
    """

    logger.info(f"Analyzing legal document with {len(request.result.pages)} pages")

    try:
        data_dict = request.model_dump()

        logger.info(f"Starting legal document analysis...")

        result = analyze_legal_document(data_dict=data_dict)

        logger.info(f"Analysis complete. Keywords: {result['keywords']}")

        # check number of keywords, construct expert query
        keywords = result["keywords"]  # list of lists keywords

        queries = []
        # Build expert query from keywords
        for keywords in result["keywords"]:
            expert_query = keywords[0]
            rest_query = " and ".join(keywords[1:])
            if rest_query:
                expert_query += f" and {rest_query}"

            logger.info(f"Searching with query: {expert_query}")

            search_request = SearchRequest(query=expert_query)
            search_response = await search_documents(search_request)
            # list of SearchResponse
            queries.append(search_response)

        # Return the analysis result with search results
        return LegalAnalysisResponse(
            summary=result["summary"],
            suggested_questions=result["suggested_questions"],
            search_responses=queries,
        )

    except Exception as e:
        logger.error(f"Legal analysis error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Legal analysis error: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search EUR-Lex documents.
    A text as input and returns a list of relevant legal documents from EUR-Lex.
    """
    if not eurlex_client:
        raise HTTPException(
            status_code=503, detail="EUR-Lex client not initialized. Check credentials."
        )

    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400, detail="Query parameter is required and cannot be empty"
        )

    # Call EUR-Lex client with default parameters
    results = eurlex_client.search_documents(
        expert_query=request.query,
        legislation=True,
        # All other parameters use their defaults:
        # page=1, page_size=10, search_language="en",
        # exclude_all_consleg=False, limit_to_latest_consleg=False
    )

    # Check if search was successful
    if not results.get("success", False):
        return SearchResponse(
            success=False,
            total_hits=0,
            num_results=0,
            documents=[],
            error=results.get("error", "Unknown error occurred"),
            query=results.get("query", ""),
        )

    # Transform results to our response format
    documents = []
    for doc in results.get("results", []):
        # Extract content
        content = doc.get("content", {})
        title = content.get("title", "No title available")
        language = content.get("language", "unknown")

        # Extract links
        links = [
            DocumentLink(type=link["type"], url=link["url"])
            for link in doc.get("document_links", [])
        ]

        # Create document object
        documents.append(
            Document(
                title=title,
                language=language,
                links=links,
                reference=doc.get("reference"),
                rank=doc.get("rank"),
            )
        )

    return SearchResponse(
        query=results.get("query", ""),
        success=True,
        total_hits=results.get("totalhits", 0),
        num_results=results.get("numhits", 0),
        documents=documents,
        error=None,
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Law Service is running",
        "status": "healthy",
        "client_initialized": eurlex_client is not None,
    }


@app.get("/languages")
async def get_languages():
    """Get list of available search languages."""
    if not eurlex_client:
        raise HTTPException(status_code=503, detail="EUR-Lex client not initialized")

    return {"languages": eurlex_client.get_available_languages()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
