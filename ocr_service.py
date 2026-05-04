import asyncio
import time
import os
import ollama
from fastapi import HTTPException
from config import settings
from logger import logger

from utils.exceptions import EmptyOutputError
from utils.image_processor import ImageProcessor


class GLMOCRService:
    """OCR service using GLM model via Ollama with streaming monitoring."""

    def __init__(self, doclayout_model=None):
        self.model = settings.OLLAMA_MODEL
        self.doclayout_model = doclayout_model

    def _call_ollama_streaming(
        self,
        tmp_path: str,
        prompt: str,
        ollama_options: dict,
    ) -> dict:
        """Blocking call to Ollama with streaming and aggressive loop monitoring."""
        tokens: list[str] = []
        stream_start = time.time()
        first_token_time = None

        stream = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [tmp_path],
                }
            ],
            options=ollama_options,
            stream=True,
            keep_alive=-1,
        )

        response_meta = None
        try:
            for chunk in stream:
                msg = getattr(chunk, "message", None)
                token = getattr(msg, "content", "") if msg else ""
                if token:
                    tokens.append(token)

                    if first_token_time is None:
                        first_token_time = time.time()
                        eval_time = round(first_token_time - stream_start, 1)
                        logger.info(
                            f"[GLM-OCR] First token  ttft={eval_time}s"
                        )

                if len(tokens) > 0 and len(tokens) % 100 == 0:
                    elapsed = round(time.time() - stream_start, 1)
                    logger.info(
                        f"[GLM-OCR] Streaming...  tokens={len(tokens)}  elapsed={elapsed}s"
                    )

                if getattr(chunk, "done", False):
                    response_meta = chunk

        except Exception as e:
            logger.error(f"[GLM-OCR] Stream error: {type(e).__name__}: {e}")
            if hasattr(stream, "close"):
                stream.close()
            raise

        full_text = "".join(tokens)
        eval_count = getattr(response_meta, "eval_count", 0) or len(tokens)

        if eval_count >= settings.OLLAMA_NUM_PREDICT - 10:
            logger.warning(
                f"[GLM-OCR] Token limit hit: {eval_count}/{settings.OLLAMA_NUM_PREDICT} — output may be truncated"
            )

        return {
            "text": full_text,
            "prompt_eval_count": getattr(response_meta, "prompt_eval_count", 0) or 0,
            "eval_count": eval_count,
        }

    async def _process_single(
        self,
        tmp_path: str,
        original_size: tuple[int, int],
        filename: str,
        prompt: str,
        start_time: float,
    ) -> dict:
        """Single OCR attempt with given prompt."""
        ollama_options = {
            "temperature": settings.OLLAMA_TEMPERATURE,
            "num_ctx": settings.OLLAMA_NUM_CTX,
            "num_predict": settings.OLLAMA_NUM_PREDICT,
            "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY,
            "repeat_last_n": settings.OLLAMA_REPEAT_LAST_N,
            "top_k": settings.OLLAMA_TOP_K,
            "top_p": settings.OLLAMA_TOP_P,
        }

        ollama_start = time.time()
        logger.info(
            f"[GLM-OCR] START  file={filename}  model={self.model}  max_size={max(original_size)}px  timeout={settings.OLLAMA_TIMEOUT}s"
        )

        result = await asyncio.wait_for(
            asyncio.to_thread(
                self._call_ollama_streaming,
                tmp_path,
                prompt,
                ollama_options,
            ),
            timeout=settings.OLLAMA_TIMEOUT,
        )

        ollama_time = round(time.time() - ollama_start, 3)

        # No bbox extraction for GLM model
        result_text = result["text"].strip()
        bboxes = []
        prompt_tokens = result["prompt_eval_count"]
        response_tokens = result["eval_count"]
        total_tokens = prompt_tokens + response_tokens
        total_time = round(time.time() - start_time, 3)
        token_speed = round(response_tokens / ollama_time, 1) if ollama_time > 0 else 0

        logger.info(
            f"[GLM-OCR] DONE   file={filename}  total={total_time}s  infer={ollama_time}s  "
            f"tokens={prompt_tokens}p+{response_tokens}r={total_tokens}  speed={token_speed}tok/s  output={len(result_text)}chars"
        )

        if not result_text or (len(result_text) < 20 and response_tokens < 50):
            logger.warning(
                f"[GLM-OCR] EMPTY  file={filename}  chars={len(result_text)}  tokens={response_tokens}  likely hallucination"
            )
            raise EmptyOutputError(
                f"Output too short or empty: {len(result_text)} chars, "
                f"{response_tokens} tokens"
            )

        return {
            "text": result_text,
            "bboxes": bboxes,
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "processing_time": f"{total_time}s",
            "ollama_time": f"{ollama_time}s",
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "total_tokens": total_tokens,
        }

    async def process(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
    ) -> dict:
        """Process an image through GLM OCR with smart fallback strategies."""
        start_time = time.time()
        last_error = None

        logger.info(f"[GLM-OCR] QUEUE  file={filename}  strategies={len(ImageProcessor.RETRY_SIZES)}")

        strategies = [
            {"max_size": size, "prompt": prompt} for size in ImageProcessor.RETRY_SIZES
        ]

        for attempt, strategy in enumerate(strategies, start=1):
            max_size = strategy["max_size"]
            current_prompt = strategy["prompt"]
            tmp_path = None

            try:
                try:
                    tmp_path, original_size, _ = await asyncio.to_thread(
                        ImageProcessor.preprocess_image,
                        file_content,
                        max_size,
                        attempt > 1,
                    )
                except ValueError as e:
                    logger.warning(
                        f"[GLM-OCR] REJECT file={filename}  reason={e}"
                    )
                    raise HTTPException(status_code=400, detail=str(e))

                saved_size = os.path.getsize(tmp_path) / 1024
                logger.info(
                    f"[GLM-OCR] IMG    file={filename}  attempt={attempt}/{len(strategies)}  "
                    f"orig={original_size[0]}x{original_size[1]}  target={max_size}px  disk={saved_size:.0f}KB"
                )

                result = await self._process_single(
                    tmp_path,
                    original_size,
                    filename,
                    current_prompt,
                    start_time,
                )

                if attempt > 1:
                    result["retried"] = True
                    result["retry_attempt"] = attempt
                    result["retry_strategy"] = (
                        f"size:{max_size}px,prompt:{current_prompt}"
                    )

                return result

            except (asyncio.TimeoutError, EmptyOutputError) as e:
                last_error = e
                logger.warning(
                    f"[GLM-OCR] RETRY  file={filename}  attempt={attempt}/{len(strategies)}  error={type(e).__name__}"
                )

            except HTTPException:
                raise

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        self._raise_http_error(last_error, filename, start_time)

    @staticmethod
    def _raise_http_error(error: Exception, filename: str, start_time: float) -> None:
        """Convert exception to appropriate HTTPException."""
        elapsed = round(time.time() - start_time, 3)

        if isinstance(error, asyncio.TimeoutError):
            logger.error(
                f"[GLM-OCR] TIMEOUT file={filename}  elapsed={elapsed}s  limit={settings.OLLAMA_TIMEOUT}s"
            )
            raise HTTPException(
                status_code=504,
                detail=(
                    f"OCR timed out after {elapsed}s. "
                    f"Try a smaller image or 'Free OCR.' prompt."
                ),
            )

        if isinstance(error, EmptyOutputError):
            logger.warning(
                f"[GLM-OCR] EMPTY  file={filename}  elapsed={elapsed}s"
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"OCR returned empty/minimal output after {elapsed}s. "
                    f"The image may be too noisy for OCR."
                ),
            )

        if isinstance(error, HTTPException):
            raise error

        logger.error(
            f"[GLM-OCR] ERROR  file={filename}  elapsed={elapsed}s  {type(error).__name__}: {error}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {type(error).__name__}: {error}",
        )
