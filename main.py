from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging
from config import settings
from ocr_service import DeepSeekOCRService
from doclayout_service import DocLayoutService
from logger import logger

app = FastAPI(title="DeepSeek OCR API", version="2.3")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ocr_service = DeepSeekOCRService()
doclayout_service = DocLayoutService()


@app.post("/ocr")
@limiter.limit(settings.RATE_LIMIT)
async def deepseek_ocr(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(settings.DEFAULT_PROMPT),
    temperature: float = Form(0.0),
    num_ctx: int = Form(8192),
    num_predict: int = Form(-1),
):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=400, detail="Only image files (JPG, PNG, WebP) are supported"
        )

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    result = await ocr_service.process(
        file_content=content,
        filename=file.filename,
        prompt=prompt,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
    )

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.post("/paddle-ocr")
@limiter.limit(settings.RATE_LIMIT)
async def paddle_ocr(request: Request, file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        raise HTTPException(status_code=400, detail="Only image files are supported")

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    result = await doclayout_service.detect_figures(content, file.filename)

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "deepseek_ocr": "enabled",
        "doclayout_yolo": "enabled",
        "version": "2.3",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=3, log_level="info")
