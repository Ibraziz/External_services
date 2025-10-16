from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Law Service")


class LawRequest(BaseModel):
    text: str


class LawResponse(BaseModel):
    text: str


@app.post("/check", response_model=LawResponse)
async def check_law(request: LawRequest):
    """
    Simple law service that returns the input text as-is.
    """
    return LawResponse(text=request.text)


@app.get("/")
async def root():
    return {"message": "Law Service is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
