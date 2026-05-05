import asyncio
import concurrent.futures
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import ollama as ollama_client

from app.core.config import settings
from app.core.logging import logger
from app.api.routes import router


async def _keepalive_loop():
    models = [settings.OLLAMA_MODEL, settings.OLLAMA_TRANSLATE_MODEL]
    while True:
        try:
            await asyncio.sleep(settings.OLLAMA_KEEPALIVE_INTERVAL)
            ps_response = await asyncio.to_thread(ollama_client.ps)
            running = [m.model for m in getattr(ps_response, "models", [])]

            for model in models:
                if any(model in n for n in running):
                    logger.info(f"[Keepalive] {model} OK")
                else:
                    logger.warning(f"[Keepalive] Reloading {model}")
                    await asyncio.to_thread(
                        ollama_client.generate, model=model, prompt="", keep_alive=-1,
                        options={"num_ctx": settings.OLLAMA_NUM_CTX}
                    )
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"[Keepalive] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models = [settings.OLLAMA_MODEL, settings.OLLAMA_TRANSLATE_MODEL]
    logger.info("[Startup] Checking Ollama...")

    try:
        ps_response = await asyncio.wait_for(asyncio.to_thread(ollama_client.ps), timeout=10)
        running = [m.model for m in getattr(ps_response, "models", [])]

        for model in models:
            if any(model in n for n in running):
                logger.info(f"[Startup] {model} loaded")
            else:
                logger.info(f"[Startup] Loading {model}...")
                await asyncio.wait_for(
                    asyncio.to_thread(ollama_client.generate, model=model, prompt="", keep_alive=-1),
                    timeout=180
                )
    except Exception as e:
        logger.warning(f"[Startup] Ollama check: {e}")

    task = asyncio.create_task(_keepalive_loop())
    logger.info("[Startup] API ready")
    yield
    task.cancel()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.API_TITLE,
        version=settings.API_VERSION,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="")
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    loop = asyncio.new_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    loop.set_default_executor(executor)
    asyncio.set_event_loop(loop)

    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, workers=1, log_level="warning", access_log=False)
