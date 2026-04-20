import asyncio
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from config import settings
from ocr_service import DeepSeekOCRService
from doclayout_service import DocLayoutService
from logger import logger


# --- Services ---
ocr_service = DeepSeekOCRService()
doclayout_service = DocLayoutService()


# --- Background keepalive ---
async def _ollama_keepalive_loop():
    """Ping Ollama every N seconds to keep model loaded and verify responsiveness."""
    import ollama as ollama_client

    while True:
        try:
            await asyncio.sleep(settings.OLLAMA_KEEPALIVE_INTERVAL)
            start = time.time()
            await asyncio.to_thread(
                ollama_client.chat,
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                keep_alive=-1,
            )
            elapsed = round(time.time() - start, 1)
            logger.info(f"[keepalive] Ollama OK ({elapsed}s)")
        except asyncio.CancelledError:
            logger.info("[keepalive] Task cancelled")
            break
        except Exception as e:
            logger.error(
                f"[keepalive] Ollama FAILED: {type(e).__name__}: {e}"
            )


# --- Lifespan (startup + shutdown) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warmup model + start keepalive. Shutdown: cancel keepalive."""
    import ollama as ollama_client

    # Warmup — pre-load model into memory (with timeout)
    logger.info("[lifespan] Warming up Ollama model...")
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                ollama_client.chat,
                model=settings.OLLAMA_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                keep_alive=-1,
            ),
            timeout=120,  # Max 2 phút cho warmup, nếu lâu hơn = skip
        )
        logger.info("[lifespan] Ollama model loaded and ready")
    except asyncio.TimeoutError:
        logger.warning(
            "[lifespan] Warmup timed out after 120s — model may still be loading. "
            "App will start, keepalive task will retry."
        )
    except Exception as e:
        logger.warning(
            f"[lifespan] Warmup failed (non-fatal): {type(e).__name__}: {e}"
        )

    # Start background keepalive
    keepalive_task = asyncio.create_task(_ollama_keepalive_loop())
    logger.info(
        f"[lifespan] Keepalive started "
        f"(interval: {settings.OLLAMA_KEEPALIVE_INTERVAL}s)"
    )

    yield

    # Shutdown
    keepalive_task.cancel()
    try:
        await keepalive_task
    except asyncio.CancelledError:
        pass
    logger.info("[lifespan] Shutdown complete")


# --- App ---
app = FastAPI(title="DeepSeek OCR API", version="2.4", lifespan=lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ocr")
@limiter.limit(settings.RATE_LIMIT)
async def deepseek_ocr(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(settings.DEFAULT_PROMPT),
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
    """Health check with Ollama model status."""
    ollama_status = "unknown"
    try:
        import ollama as ollama_client
        ps_response = await asyncio.to_thread(ollama_client.ps)
        running_models = [
            m.model for m in getattr(ps_response, "models", [])
        ]
        if any(settings.OLLAMA_MODEL in name for name in running_models):
            ollama_status = "model_loaded"
        else:
            ollama_status = "model_not_loaded"
    except Exception as e:
        ollama_status = f"error: {type(e).__name__}"

    return {
        "status": "ok",
        "deepseek_ocr": "enabled",
        "doclayout_yolo": "enabled",
        "ollama_status": ollama_status,
        "version": "2.4",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
