from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.responses import JSONResponse
from openai import OpenAI
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Translation Service")

# Connect to local vLLM server
client = OpenAI(
    api_key="dummy",
    base_url="http://localhost:8080/v1"
)


class TranslationRequest(BaseModel):
    text: str
    target_language: str


class TranslationResponse(BaseModel):
    translated_text: str

@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """Translation service using Gemma-3-12b"""
    
    logger.info(f"Translation request: '{request.text}' -> {request.target_language}")
    
    try:
        # MUCH SIMPLER PROMPT - conversational style
        prompt = f"Translate to {request.target_language}: {request.text}"
        
        logger.info(f"Sending prompt: {prompt}")

        response = client.chat.completions.create(
            model="google/gemma-2-9b-it",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        logger.info(f"Raw response: {response}")
        
        if not response.choices or not response.choices[0].message.content:
            logger.error("Empty response from model!")
            raise HTTPException(status_code=500, detail="Model returned empty response")

        translated_text = response.choices[0].message.content.strip()
        logger.info(f"Translated: '{translated_text}'")

        return JSONResponse(
            content={"translated_text": translated_text},
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Translation Service is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)