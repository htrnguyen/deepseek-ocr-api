import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from config import settings
from ocr_service import GLMOCRService
from doclayout_service import DocLayoutService
from paddle_detect_service import PaddleDetectService
from logger import logger
import ollama as ollama_client
from pydantic import BaseModel
from translate_service import TranslateService
from chandra_service import ChandraService


doclayout_service = DocLayoutService()
paddle_detect_service = PaddleDetectService()
ocr_service = GLMOCRService()
translate_service = TranslateService()
chandra_service = ChandraService()


class TranslateRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "auto"


async def validate_image_upload(file: UploadFile = File(...)) -> UploadFile:
    """Dependency to validate uploaded image type and size."""
    if file.content_type not in {"image/jpeg", "image/jpg", "image/png", "image/webp"}:
        raise HTTPException(
            status_code=400, detail="Only image files (JPG, PNG, WebP) are supported"
        )

    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        max_mb = settings.MAX_FILE_SIZE // (1024 * 1024)
        raise HTTPException(status_code=400, detail=f"File too large (max {max_mb}MB)")

    file.file_content = content
    return file


async def _ollama_keepalive_loop():
    """Ping Ollama every N seconds to verify models are still loaded."""

    models_to_check = [settings.OLLAMA_MODEL, settings.OLLAMA_TRANSLATE_MODEL]

    while True:
        try:
            await asyncio.sleep(settings.OLLAMA_KEEPALIVE_INTERVAL)
            start = time.time()
            ps_response = await asyncio.to_thread(ollama_client.ps)
            running = [m.model for m in getattr(ps_response, "models", [])]
            elapsed = round(time.time() - start, 1)

            for model_name in models_to_check:
                if any(model_name in name for name in running):
                    logger.info(
                        f"[_ollama_keepalive_loop] Ollama OK | Model {model_name} loaded | Time: {elapsed}s"
                    )
                else:
                    logger.warning(
                        f"[_ollama_keepalive_loop] Model {model_name} not loaded | Reloading..."
                    )
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            ollama_client.generate,
                            model=model_name,
                            prompt="",
                            keep_alive=-1,
                            options={"num_ctx": settings.OLLAMA_NUM_CTX},
                        ),
                        timeout=180,
                    )
                    logger.info(
                        f"[_ollama_keepalive_loop] Model {model_name} reloaded successfully"
                    )
        except asyncio.CancelledError:
            logger.info("[_ollama_keepalive_loop] Task cancelled")
            break
        except Exception as e:
            logger.error(
                f"[_ollama_keepalive_loop] Ollama FAILED | Error: {type(e).__name__}: {e}"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify models + start keepalive. Shutdown: cancel keepalive."""

    models_to_check = [settings.OLLAMA_MODEL, settings.OLLAMA_TRANSLATE_MODEL]
    logger.info("[lifespan] Checking Ollama status...")
    try:
        ps_response = await asyncio.wait_for(
            asyncio.to_thread(ollama_client.ps),
            timeout=10,
        )
        running = [m.model for m in getattr(ps_response, "models", [])]

        for model_name in models_to_check:
            if any(model_name in name for name in running):
                logger.info(
                    f"[lifespan] Ollama ready | Model '{model_name}' loaded in GPU"
                )
            else:
                logger.warning(
                    f"[lifespan] Model '{model_name}' not in memory | "
                    f"Loading... (this may take 1-2 min on first run)"
                )
                await asyncio.wait_for(
                    asyncio.to_thread(
                        ollama_client.generate,
                        model=model_name,
                        prompt="",
                        keep_alive=-1,
                        options={"num_ctx": settings.OLLAMA_NUM_CTX},
                    ),
                    timeout=180,
                )
                logger.info(f"[lifespan] Model {model_name} loaded successfully")
    except asyncio.TimeoutError:
        logger.warning("[lifespan] Ollama check timed out | App will start anyway")
    except Exception as e:
        logger.warning(
            f"[lifespan] Ollama check failed (non-fatal) | Error: {type(e).__name__}: {e}"
        )

    keepalive_task = asyncio.create_task(_ollama_keepalive_loop())
    logger.info(
        f"[lifespan] Keepalive started | "
        f"Interval: {settings.OLLAMA_KEEPALIVE_INTERVAL}s"
    )

    logger.info("[lifespan] Preloading Chandra model for maximum performance...")
    try:
        await asyncio.to_thread(chandra_service.load)
    except Exception as e:
        logger.error(f"[lifespan] Failed to preload Chandra model: {e}")

    yield

    keepalive_task.cancel()
    try:
        await keepalive_task
    except asyncio.CancelledError:
        pass
    logger.info("[lifespan] Shutdown complete")


app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Unified AI Service Aggregator optimized for Apple Silicon (Mac M-Series). Integrates GLM, Chandra, YOLO, and Gemma models for OCR, layout detection, and translation.",
    lifespan=lifespan,
)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.post(
    "/ocr",
    tags=["1. OCR & Extraction"],
    summary="GLM OCR (Ollama)",
    description="Perform text extraction on an image using the GLM-OCR model via Ollama. It is optimized to output clean Markdown while preserving document structure.",
)
@limiter.limit(settings.RATE_LIMIT)
async def glm_ocr(
    request: Request,
    file: UploadFile = Depends(validate_image_upload),
    prompt: str = Form(settings.DEFAULT_PROMPT),
):
    result = await ocr_service.process(
        file_content=file.file_content,
        filename=file.filename,
        prompt=prompt,
    )

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.post(
    "/paddle-detect",
    tags=["2. Layout & Vision"],
    summary="Paddle Detect",
    description="Lightweight bounding box and text detection using the PaddleOCR engine. Good for quick text coordinate extraction.",
)
@limiter.limit(settings.RATE_LIMIT)
async def paddle_detect(
    request: Request, file: UploadFile = Depends(validate_image_upload)
):
    result = await paddle_detect_service.detect(file.file_content, file.filename)

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.post(
    "/doclayout",
    tags=["2. Layout & Vision"],
    summary="YOLO DocLayout",
    description="Ultra-fast structural detection using a YOLO model. Extracts bounding boxes for figures, tables, headers, and text blocks. Does NOT extract text content.",
)
@limiter.limit(settings.RATE_LIMIT)
async def doclayout_detect(
    request: Request, file: UploadFile = Depends(validate_image_upload)
):
    result = await doclayout_service.detect_figures(file.file_content, file.filename)

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.post(
    "/paddle-ocr",
    tags=["2. Layout & Vision"],
    summary="Paddle OCR (Legacy Alias)",
    description="Legacy alias for /doclayout. Performs YOLO structural detection.",
)
@limiter.limit(settings.RATE_LIMIT)
async def paddle_ocr_legacy(
    request: Request, file: UploadFile = Depends(validate_image_upload)
):
    result = await doclayout_service.detect_figures(file.file_content, file.filename)

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.post(
    "/translate",
    tags=["3. NLP Services"],
    summary="Translate (Gemma)",
    description="Translate text using the translategemma model. Optimized to strictly preserve LaTeX math equations ($, $$) and complex Markdown formatting from OCR outputs.",
)
@limiter.limit(settings.RATE_LIMIT)
async def translate_text(request: Request, data: TranslateRequest):
    result = await translate_service.translate(
        text=data.text,
        source_language=data.source_language,
        target_language=data.target_language,
    )
    return {"success": True, **result, "status": "success"}


@app.post(
    "/chandra-ocr",
    tags=["1. OCR & Extraction"],
    summary="Chandra OCR (MPS)",
    description="Advanced layout detection and high-fidelity Markdown parsing using the Chandra model. Lazily loaded and optimized to run on Mac's Unified Memory (MPS).",
)
@limiter.limit(settings.RATE_LIMIT)
async def chandra_ocr(
    request: Request, file: UploadFile = Depends(validate_image_upload)
):
    result = await chandra_service.process(file.file_content, file.filename)
    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.get(
    "/health",
    tags=["4. System"],
    summary="System Health & Status",
    description="Returns the operational status of all registered AI modules and currently loaded Ollama models in VRAM.",
)
async def health():
    ollama_status = {}
    try:
        ps_response = await asyncio.to_thread(ollama_client.ps)
        running_models = [m.model for m in getattr(ps_response, "models", [])]

        for model_name in [settings.OLLAMA_MODEL, settings.OLLAMA_TRANSLATE_MODEL]:
            if any(model_name in name for name in running_models):
                ollama_status[model_name] = "loaded"
            else:
                ollama_status[model_name] = "not_loaded"
    except Exception as e:
        ollama_status = f"error: {type(e).__name__}"

    return {
        "status": "ok",
        "glm_ocr": "enabled",
        "chandra_ocr": "enabled",
        "doclayout_yolo": "enabled",
        "paddle_detect": "enabled",
        "translate": "enabled",
        "ollama_status": ollama_status,
        "version": settings.API_VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host=settings.HOST, port=settings.PORT, workers=1, log_level="info"
    )
