from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from fastapi.responses import JSONResponse
from openai import OpenAI


app = FastAPI(title="Translation Service")


client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


class TranslationRequest(BaseModel):
    text: str
    target_language: str


class TranslationResponse(BaseModel):
    translated_text: str


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translation service that uses Gemini Flash model to translate text.

    Args:
        request: Contains text to translate and target_language

    Returns:
        TranslationResponse with translated text
    """
    try:
        # Create the translation prompt
        prompt = f"Translate the following text to {request.target_language}. Only provide the translation, no explanations or additional text:\n\n{request.text}"

        # Call Gemini Flash model via OpenAI library
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "You are a helpful translator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  
        )

        # Extract the translated text
        translated_text = response.choices[0].message.content.strip()

        return JSONResponse(
            content={"translated_text": translated_text},
            media_type="application/json; charset=utf-8"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Translation Service is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
