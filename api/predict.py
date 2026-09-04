import os
import requests

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

HF_API_URL = os.environ.get("HF_API_URL")


@app.post("/")
async def predict(file: UploadFile = File(...)):
    if not HF_API_URL:
        raise HTTPException(
            status_code=500,
            detail="HF_API_URL environment variable is not configured."
        )

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Invalid file."
        )

    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/webp"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG, PNG and WEBP images are supported."
        )

    image_bytes = await file.read()

    try:
        response = requests.post(
            HF_API_URL,
            files={
                "file": (
                    file.filename,
                    image_bytes,
                    file.content_type
                )
            },
            timeout=60
        )

        response.raise_for_status()

        return JSONResponse(
            content=response.json()
        )

    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Inference server error: {str(e)}"
        )