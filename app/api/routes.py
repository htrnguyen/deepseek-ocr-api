import asyncio
import time
from fastapi import APIRouter, UploadFile, File, Form, Request, Depends
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
import ollama as ollama_client

from app.core.config import settings
from app.core.logging import logger
from app.schemas.requests import TranslateRequest
from app.schemas.responses import HealthResponse
from app.services import OCRService, TranslateService, LayoutService, DetectionService, OcrLayoutService

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

ocr_service = OCRService()
translate_service = TranslateService()
layout_service = LayoutService()
detection_service = DetectionService()
ocr_layout_service = OcrLayoutService()


async def validate_image(file: UploadFile = File(...)) -> UploadFile:
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        return JSONResponse(status_code=400, content={"detail": "Only JPG, PNG, WebP supported"})

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        return JSONResponse(status_code=400, content={"detail": f"File too large (max {max_mb}MB)"})

    file.file_content = content
    return file


@router.post("/ocr")
@limiter.limit(settings.RATE_LIMIT)
async def ocr_endpoint(request: Request, file: UploadFile = Depends(validate_image), prompt: str = Form(settings.DEFAULT_PROMPT)):
    logger.info(f"[Endpoint] POST /ocr file={file.filename}")
    result = await ocr_service.process(file.file_content, file.filename, prompt)
    return result


@router.post("/ocr-layout")
@limiter.limit(settings.RATE_LIMIT)
async def ocr_layout_endpoint(request: Request, file: UploadFile = Depends(validate_image)):
    logger.info(f"[Endpoint] POST /ocr-layout file={file.filename}")
    result = await ocr_layout_service.process(file.file_content, file.filename)
    return result


@router.post("/paddle-detect")
@limiter.limit(settings.RATE_LIMIT)
async def detect_endpoint(request: Request, file: UploadFile = Depends(validate_image)):
    logger.info(f"[Endpoint] POST /paddle-detect file={file.filename}")
    result = await detection_service.process(file.file_content, file.filename)
    return result


@router.post("/doclayout")
@limiter.limit(settings.RATE_LIMIT)
async def layout_endpoint(request: Request, file: UploadFile = Depends(validate_image)):
    logger.info(f"[Endpoint] POST /doclayout file={file.filename}")
    result = await layout_service.process(file.file_content, file.filename)
    return result


@router.post("/paddle-ocr")
@limiter.limit(settings.RATE_LIMIT)
async def layout_legacy_endpoint(request: Request, file: UploadFile = Depends(validate_image)):
    logger.info(f"[Endpoint] POST /paddle-ocr file={file.filename}")
    result = await layout_service.process(file.file_content, file.filename)
    return result


@router.post("/translate")
@limiter.limit(settings.RATE_LIMIT)
async def translate_endpoint(request: Request, data: TranslateRequest):
    logger.info(f"[Endpoint] POST /translate text_len={len(data.text)}")
    result = await translate_service.process(data.text, data.source_language, data.target_language)
    return result


@router.get("/health")
async def health_endpoint():
    ollama_status = {}
    try:
        ps_response = await asyncio.to_thread(ollama_client.ps)
        running = [m.model for m in getattr(ps_response, "models", [])]
        for name in [settings.OLLAMA_MODEL, settings.OLLAMA_TRANSLATE_MODEL]:
            ollama_status[name] = "loaded" if any(name in n for n in running) else "not_loaded"
    except Exception as e:
        ollama_status = f"error: {type(e).__name__}"

    return HealthResponse(ollama_status=ollama_status)
