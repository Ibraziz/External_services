from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Translation Service")


class TranslationRequest(BaseModel):
    text: str


class TranslationResponse(BaseModel):
    text: str


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Simple translation service that returns the input text as-is.
    """
    return TranslationResponse(text=request.text)


@app.get("/")
async def root():
    return {"message": "Translation Service is running"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
