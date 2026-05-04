import time
import asyncio
import ollama
from fastapi import HTTPException
from config import settings
from logger import logger


class TranslateService:
    """Service for translating text using translategemma model via Ollama."""

    _CONCURRENCY = getattr(settings, "OLLAMA_TRANSLATE_CONCURRENCY", 4)

    def __init__(self):
        self.model = settings.OLLAMA_TRANSLATE_MODEL
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._CONCURRENCY)
        return self._semaphore

    async def translate(
        self, text: str, source_language: str, target_language: str
    ) -> dict:
        """Translate text to the target language."""
        start_time = time.time()
        char_count = len(text)
        logger.info(
            f"[Translate] START  lang={source_language}->{target_language}  input={char_count} chars"
        )

        instruction = (
            "IMPORTANT: Strictly preserve all Markdown formatting, structural layout, "
            "and LaTeX math equations (like $...$ or $$...$$) exactly as they appear in the original text."
        )

        if not source_language or source_language.lower() == "auto":
            prompt = f"Translate the following text to {target_language}. {instruction}\n\n{text}"
        else:
            prompt = f"Translate the following text from {source_language} to {target_language}. {instruction}\n\n{text}"

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
            result_text = getattr(response, "response", None) or ""
            if isinstance(result_text, str):
                result_text = result_text.strip()
            eval_count = getattr(response, "eval_count", 0) or 0

            logger.info(
                f"[Translate] DONE   lang={source_language}->{target_language}  "
                f"time={elapsed}s  tokens={eval_count}  output={len(result_text)} chars"
            )

            return {
                "translated_text": result_text,
                "processing_time": f"{elapsed}s",
                "tokens": eval_count,
            }
        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            logger.error(
                f"[Translate] ERROR  time={elapsed}s  {type(e).__name__}: {e}"
            )
            raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
