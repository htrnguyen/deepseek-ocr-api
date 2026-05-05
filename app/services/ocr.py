import asyncio
import time
import os
import ollama
from fastapi import HTTPException
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BaseService
from app.utils.image import ImageProcessor


class OCRService(BaseService):
    def __init__(self):
        self.model = settings.OLLAMA_MODEL

    async def process(self, file_content: bytes, filename: str, prompt: str) -> dict:
        start = time.time()
        tmp_path = None

        try:
            tmp_path, _, _ = await asyncio.to_thread(
                ImageProcessor.preprocess, file_content
            )

            result = await self._call_ollama(tmp_path, prompt)
            total = round(time.time() - start, 2)

            return self._build_response({
                "filename": filename,
                "text": result["text"],
                "processing_time": f"{total}s",
                "tokens": result["eval_count"],
            })

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _call_ollama(self, tmp_path: str, prompt: str) -> dict:
        def _sync_call():
            tokens = []
            stream = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [tmp_path]}],
                options={
                    "temperature": settings.OLLAMA_TEMPERATURE,
                    "num_ctx": settings.OLLAMA_NUM_CTX,
                    "num_predict": settings.OLLAMA_NUM_PREDICT,
                    "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY,
                },
                stream=True,
                keep_alive=-1,
            )

            response_meta = None
            for chunk in stream:
                msg = getattr(chunk, "message", None)
                token = getattr(msg, "content", "") if msg else ""
                if token:
                    tokens.append(token)
                if getattr(chunk, "done", False):
                    response_meta = chunk

            return {
                "text": "".join(tokens),
                "eval_count": getattr(response_meta, "eval_count", len(tokens)),
            }

        return await asyncio.wait_for(
            asyncio.to_thread(_sync_call),
            timeout=settings.OLLAMA_TIMEOUT
        )
