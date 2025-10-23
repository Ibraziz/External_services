from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Translation Service")

# Connect to local vLLM server
client = OpenAI(
    # api_key="dummy",
    # base_url="http://localhost:8080/v1"
    base_url="https://router.huggingface.co/v1",
    api_key = ""
)


class TranslationRequest(BaseModel):
    text: str
    target_language: str


class TranslationResponse(BaseModel):
    translation: str
    source_language: str
    confidence: float


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    
    logger.info(f"Translation request: '{request.text}' -> {request.target_language}")
    
    try:
        
        prompt = f"Translate the following text to {request.target_language}: {request.text}"
        
        response = client.responses.parse(
            model="google/gemma-2-9b-it",
            input=[
                {"role": "system", "content": "Translate the user's text to the specified target language without adding any extra information."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            text_format = TranslationResponse
        )

        logger.info(f"Raw response: {response.output_parsed}")
        
        if not response.output_parsed:
            logger.error("Empty response from model!")
            raise HTTPException(status_code=500, detail="Model returned empty response")

        translated_text = response.output_parsed
        logger.info(f"Translated: '{translated_text.translation}'")

        response_json = jsonable_encoder({"translated_text": translated_text.translation})

        return JSONResponse(
            content=response_json
        )

    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Translation Service is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)