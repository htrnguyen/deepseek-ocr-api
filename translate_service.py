import time
import asyncio
import ollama
from fastapi import HTTPException
from config import settings
from logger import logger


class TranslateService:
    """Service for translating text using translategemma model via Ollama."""

    def __init__(self):
        self.model = settings.OLLAMA_TRANSLATE_MODEL
        self.semaphore = asyncio.Semaphore(
            getattr(settings, "OLLAMA_TRANSLATE_CONCURRENCY", 4)
        )

    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> dict:
        """Translate text to the target language."""
        start_time = time.time()
        logger.info(f"[translate] Starting translation to {target_language}")

        instruction = (
            "IMPORTANT: Strictly preserve all Markdown formatting, structural layout, "
            "and LaTeX math equations (like $...$ or $$...$$) exactly as they appear in the original text."
        )

        prompt = f"Translate the following text from {source_language} to {target_language}. {instruction}\n\n{text}"
        if not source_language or source_language.lower() == "auto":
            prompt = f"Translate the following text to {target_language}. {instruction}\n\n{text}"

        try:
            async with self.semaphore:
                response = await asyncio.to_thread(
                    ollama.generate,
                    model=self.model,
                    prompt=prompt,
                    keep_alive=-1,
                    options={
                        "temperature": settings.OLLAMA_TEMPERATURE,
                        "num_predict": settings.OLLAMA_NUM_PREDICT,
                        "num_ctx": 2048,
                    },
                )

            elapsed = round(time.time() - start_time, 3)
            result_text = response.get("response", "").strip()
            eval_count = response.get("eval_count", 0)

            logger.info(
                f"[translate] Success | Time: {elapsed}s | Tokens: {eval_count} | Output length: {len(result_text)}"
            )

            return {
                "translated_text": result_text,
                "processing_time": f"{elapsed}s",
                "tokens": eval_count,
            }
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            logger.error(
                f"[translate] Error | Time: {elapsed}s | {type(e).__name__}: {e}"
            )
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
