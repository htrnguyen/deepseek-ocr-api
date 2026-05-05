import asyncio
import ollama
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BaseService


class TranslateService(BaseService):
    def __init__(self):
        self.model = settings.OLLAMA_TRANSLATE_MODEL

    async def process(self, text: str, source_language: str, target_language: str) -> dict:
        prompt = self._build_prompt(text, source_language, target_language)

        def _sync_call():
            return ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1},
                keep_alive=-1,
            )

        result = await asyncio.to_thread(_sync_call)
        translated = result.response if hasattr(result, "response") else str(result)

        return self._build_response({
            "original": text,
            "translated": translated,
            "source_language": source_language,
            "target_language": target_language,
        })

    def _build_prompt(self, text: str, source: str, target: str) -> str:
        return f"Translate from {source} to {target}. Preserve LaTeX math ($...$ and $$...$$). Output only translation, no explanations:\n\n{text}"
