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


doclayout_service = DocLayoutService()
paddle_detect_service = PaddleDetectService()
ocr_service = GLMOCRService()


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
    """Ping Ollama every N seconds to verify model is still loaded."""

    while True:
        try:
            await asyncio.sleep(settings.OLLAMA_KEEPALIVE_INTERVAL)
            start = time.time()
            ps_response = await asyncio.to_thread(ollama_client.ps)
            running = [m.model for m in getattr(ps_response, "models", [])]
            elapsed = round(time.time() - start, 1)

            if any(settings.OLLAMA_MODEL in name for name in running):
                logger.info(
                    f"[keepalive] | Ollama OK | Model loaded | Time: {elapsed}s"
                )
            else:
                logger.warning("[keepalive] | Model not loaded | Reloading...")
                await asyncio.wait_for(
                    asyncio.to_thread(
                        ollama_client.generate,
                        model=settings.OLLAMA_MODEL,
                        prompt="",
                        keep_alive=-1,
                        options={"num_ctx": settings.OLLAMA_NUM_CTX},
                    ),
                    timeout=180,
                )
                logger.info("[keepalive] | Model reloaded successfully")
        except asyncio.CancelledError:
            logger.info("[keepalive] | Task cancelled")
            break
        except Exception as e:
            logger.error(
                f"[keepalive] | Ollama FAILED | Error: {type(e).__name__}: {e}"
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: verify model + start keepalive. Shutdown: cancel keepalive."""

    logger.info("[lifespan] | Checking Ollama status...")
    try:
        ps_response = await asyncio.wait_for(
            asyncio.to_thread(ollama_client.ps),
            timeout=10,
        )
        running = [m.model for m in getattr(ps_response, "models", [])]
        if any(settings.OLLAMA_MODEL in name for name in running):
            logger.info(
                f"[lifespan] | Ollama ready | Model '{settings.OLLAMA_MODEL}' loaded in GPU"
            )
        else:
            logger.warning(
                f"[lifespan] | Model '{settings.OLLAMA_MODEL}' not in memory | "
                f"Loading... (this may take 1-2 min on first run)"
            )
            await asyncio.wait_for(
                asyncio.to_thread(
                    ollama_client.generate,
                    model=settings.OLLAMA_MODEL,
                    prompt="",
                    keep_alive=-1,
                    options={"num_ctx": settings.OLLAMA_NUM_CTX},
                ),
                timeout=180,
            )
            logger.info("[lifespan] | Model loaded successfully")
    except asyncio.TimeoutError:
        logger.warning("[lifespan] | Ollama check timed out | App will start anyway")
    except Exception as e:
        logger.warning(
            f"[lifespan] | Ollama check failed (non-fatal) | Error: {type(e).__name__}: {e}"
        )

    keepalive_task = asyncio.create_task(_ollama_keepalive_loop())
    logger.info(
        f"[lifespan] | Keepalive started | "
        f"Interval: {settings.OLLAMA_KEEPALIVE_INTERVAL}s"
    )

    yield

    keepalive_task.cancel()
    try:
        await keepalive_task
    except asyncio.CancelledError:
        pass
    logger.info("[lifespan] | Shutdown complete")


app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION, lifespan=lifespan)
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


@app.post("/ocr")
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


@app.post("/paddle-ocr")
@limiter.limit(settings.RATE_LIMIT)
async def paddle_ocr(
    request: Request, file: UploadFile = Depends(validate_image_upload)
):
    result = await doclayout_service.detect_figures(file.file_content, file.filename)

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.post("/paddle-detect")
@limiter.limit(settings.RATE_LIMIT)
async def paddle_detect(
    request: Request, file: UploadFile = Depends(validate_image_upload)
):
    result = await paddle_detect_service.detect(file.file_content, file.filename)

    return {"success": True, "filename": file.filename, **result, "status": "success"}


@app.get("/health")
async def health():
    """Health check with Ollama model status."""
    ollama_status = "unknown"
    try:

        ps_response = await asyncio.to_thread(ollama_client.ps)
        running_models = [m.model for m in getattr(ps_response, "models", [])]
        if any(settings.OLLAMA_MODEL in name for name in running_models):
            ollama_status = "model_loaded"
        else:
            ollama_status = "model_not_loaded"
    except Exception as e:
        ollama_status = f"error: {type(e).__name__}"

    return {
        "status": "ok",
        "glm_ocr": "enabled",
        "doclayout_yolo": "enabled",
        "paddle_detect": "enabled",
        "ollama_status": ollama_status,
        "version": settings.API_VERSION,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host=settings.HOST, port=settings.PORT, workers=1, log_level="info"
    )
