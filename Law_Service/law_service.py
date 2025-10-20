from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from eurlex_client import EURLexClient
from openai import OpenAI
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Law Service",
    description="RESTful API wrapper for EUR-Lex document search"
)

# Initialize EUR-Lex client (credentials loaded from .env)
try:
    eurlex_client = EURLexClient()
except ValueError as e:
    print(f"Warning: EUR-Lex client initialization failed: {e}")
    eurlex_client = None


# Request/Response Models
class SearchRequest(BaseModel):
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "QUICK_SEARCH ~ transport"
            }
        }


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
    success: bool
    total_hits: int
    num_results: int
    documents: List[Document]
    error: Optional[str] = None


# Connect to local vLLM server
client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8080/v1"
)

class LawkeywordsRequest(BaseModel):
    text: str

class LawkeywordsResponse(BaseModel):
    keywords: str

@app.post("/Lawkeywords", response_model=LawkeywordsResponse)
async def extract_law_keywords(request: LawkeywordsRequest):
    """relevant keywords service using Gemma-2-9b"""
    
    logger.info(f"Extracting law keywords from text: '{request.text[:100]}...'")
    
    try:
        # MUCH SIMPLER PROMPT - conversational style
        prompt = f"Given the following contract, identify the 3 most relevant law keywords as single words, no extra information: {request.text}"
        
        logger.info(f"Sending prompt: {prompt}")

        response = client.chat.completions.create(
            model="google/gemma-2-9b-it",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        logger.info(f"Raw response: {response}")
        
        if not response.choices or not response.choices[0].message.content:
            logger.error("Empty response from model!")
            raise HTTPException(status_code=500, detail="Model returned empty response")

        keywords_text = response.choices[0].message.content.strip()
        logger.info(f"Keywords: '{keywords_text}'")

        return LawkeywordsResponse(keywords=keywords_text)

    except Exception as e:
        logger.error(f"Keyword extraction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Keyword extraction error: {str(e)}")




# Endpoints
@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search EUR-Lex documents using expert query syntax.
    
    Example queries:
    - `QUICK_SEARCH ~ transport`
    - `QUICK_SEARCH ~ "non-disclosure" AND QUICK_SEARCH ~ agreement`
    - `QUICK_SEARCH ~ "air transport"`
    """
    if not eurlex_client:
        raise HTTPException(
            status_code=503,
            detail="EUR-Lex client not initialized. Check credentials."
        )
    
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query parameter is required and cannot be empty"
        )
    
    # Call EUR-Lex client with default parameters
    results = eurlex_client.search_documents(
        expert_query=request.query
        # All other parameters use their defaults:
        # page=1, page_size=10, search_language="en",
        # exclude_all_consleg=False, limit_to_latest_consleg=False
    )
    
    # Check if search was successful
    if not results.get('success', False):
        return SearchResponse(
            success=False,
            total_hits=0,
            num_results=0,
            documents=[],
            error=results.get('error', 'Unknown error occurred')
        )
    
    # Transform results to our response format
    documents = []
    for doc in results.get('results', []):
        # Extract content
        content = doc.get('content', {})
        title = content.get('title', 'No title available')
        language = content.get('language', 'unknown')
        
        # Extract links
        links = [
            DocumentLink(type=link['type'], url=link['url'])
            for link in doc.get('document_links', [])
        ]
        
        # Create document object
        documents.append(Document(
            title=title,
            language=language,
            links=links,
            reference=doc.get('reference'),
            rank=doc.get('rank')
        ))
    
    return SearchResponse(
        success=True,
        total_hits=results.get('totalhits', 0),
        num_results=results.get('numhits', 0),
        documents=documents,
        error=None
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Law Service is running",
        "status": "healthy",
        "client_initialized": eurlex_client is not None
    }


@app.get("/languages")
async def get_languages():
    """Get list of available search languages."""
    if not eurlex_client:
        raise HTTPException(
            status_code=503,
            detail="EUR-Lex client not initialized"
        )
    
    return {
        "languages": eurlex_client.get_available_languages()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)