import asyncio
import time
import os
import ollama
from fastapi import HTTPException
from config import settings
from logger import logger

from utils.exceptions import TokenLoopError, EmptyOutputError
from utils.loop_detector import LoopDetector
from utils.post_processor import OCRPostProcessor
from utils.image_processor import ImageProcessor


class DeepSeekOCRService:
    """OCR service using DeepSeek model via Ollama with streaming monitoring."""

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

        response_meta = {}
        try:
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    tokens.append(token)

                    if first_token_time is None:
                        first_token_time = time.time()
                        eval_time = round(first_token_time - stream_start, 1)
                        logger.info(
                            f"[streaming] | First token | Time: {eval_time}s (prompt eval)"
                        )

                if len(tokens) > 0 and len(tokens) % 50 == 0:
                    elapsed = round(time.time() - stream_start, 1)
                    preview = "".join(tokens[-20:])[:80]
                    logger.info(
                        f"[streaming] | Progress: {len(tokens)} tokens | "
                        f"Time: {elapsed}s | Last: '{preview}'"
                    )

                if len(tokens) >= 20 and len(tokens) % 10 == 0:
                    if LoopDetector.detect(tokens):
                        elapsed = round(time.time() - stream_start, 1)
                        full_output = "".join(tokens)
                        logger.warning(
                            f"[loop_detect] ═══ LOOP DETECTED ═══\n"
                            f"  Prompt: '{prompt}'\n"
                            f"  Tokens: {len(tokens)} | Elapsed: {elapsed}s\n"
                            f"  Full output ({len(full_output)} chars):\n"
                            f"  ──────────────────────────────────────\n"
                            f"  {full_output[:500]}\n"
                            f"  ──────────────────────────────────────"
                        )
                        if hasattr(stream, "close"):
                            stream.close()

                        valid_tokens = tokens[:-100] if len(tokens) > 100 else []
                        if len(valid_tokens) >= 50:
                            logger.info(
                                f"[loop_detect] | Partial Success | Recovered {len(valid_tokens)} "
                                f"valid tokens before the loop"
                            )
                            tokens = valid_tokens
                            response_meta["eval_count"] = len(tokens)
                            break
                        else:
                            raise TokenLoopError(
                                f"Loop after {len(tokens)} tokens ({elapsed}s). "
                                f"Last: '{''.join(tokens[-8:])[:60]}...'"
                            )

                if chunk.get("done"):
                    response_meta = chunk

        except ollama.ResponseError as e:
            if "looping" in str(e).lower():
                elapsed = round(time.time() - stream_start, 1)
                full_output = "".join(tokens)
                logger.warning(
                    f"[loop_detect] ═══ OLLAMA LOOP DETECTED ═══\n"
                    f"  Prompt: '{prompt}'\n"
                    f"  Tokens: {len(tokens)} | Elapsed: {elapsed}s\n"
                    f"  Ollama error: {e}\n"
                    f"  Output so far ({len(full_output)} chars):\n"
                    f"  ──────────────────────────────────────\n"
                    f"  {full_output[:500]}\n"
                    f"  ──────────────────────────────────────"
                )
                if hasattr(stream, "close"):
                    stream.close()

                valid_tokens = tokens[:-100] if len(tokens) > 100 else []
                if len(valid_tokens) >= 50:
                    logger.info(
                        f"[loop_detect] | Partial Success | Recovered {len(valid_tokens)} "
                        f"valid tokens before Ollama's built-in loop error"
                    )
                    tokens = valid_tokens
                    return {
                        "text": "".join(tokens),
                        "prompt_eval_count": response_meta.get("prompt_eval_count", 0)
                        or 0,
                        "eval_count": len(tokens),
                    }
                else:
                    raise TokenLoopError(
                        f"Ollama loop detection after {len(tokens)} tokens ({elapsed}s)"
                    )
            raise

        except Exception:
            if hasattr(stream, "close"):
                stream.close()
            raise

        full_text = "".join(tokens)

        eval_count = response_meta.get("eval_count", 0) or 0
        if eval_count >= settings.OLLAMA_NUM_PREDICT - 10:
            raise TokenLoopError(
                f"Hit token limit ({eval_count} >= {settings.OLLAMA_NUM_PREDICT}) "
                f"— model is stuck in an infinite loop."
            )

        return {
            "text": full_text,
            "prompt_eval_count": response_meta.get("prompt_eval_count", 0) or 0,
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

        logger.info(
            f"[process] | Ollama Config | File: {filename} | Model: {self.model} | "
            f"Prompt: '{prompt}' | Timeout: {settings.OLLAMA_TIMEOUT}s | "
            f"Repeat Penalty: {ollama_options['repeat_penalty']}"
        )

        ollama_start = time.time()
        logger.info(f"[process] | [Step 3/3] Sending streaming request to Ollama | File: {filename}")

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

        # Post-process
        result_text = OCRPostProcessor.clean(result["text"])
        prompt_tokens = result["prompt_eval_count"]
        response_tokens = result["eval_count"]
        total_tokens = prompt_tokens + response_tokens
        total_time = round(time.time() - start_time, 3)
        token_speed = round(response_tokens / ollama_time, 1) if ollama_time > 0 else 0

        logger.info(
            f"[process] | Pipeline Completed | File: {filename} | Total Time: {total_time}s | "
            f"Ollama Time: {ollama_time}s | "
            f"Tokens: {prompt_tokens}p + {response_tokens}r = {total_tokens} | "
            f"Speed: {token_speed} tok/s | Output: {len(result_text)} chars\n"
            f"  Preview: '{result_text[:200]}'"
        )

        if not result_text or (len(result_text) < 20 and response_tokens < 50):
            logger.warning(
                f"[process] | EmptyOutput | File: {filename} | {len(result_text)} chars | "
                f"{response_tokens} tokens | Likely hallucination"
            )
            raise EmptyOutputError(
                f"Output too short or empty: {len(result_text)} chars, "
                f"{response_tokens} tokens"
            )

        return {
            "text": result_text,
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
        """Process an image through DeepSeek OCR."""
        start_time = time.time()
        tmp_path = None

        logger.info(f"[process] | [Step 1/3] Pipeline Started | File: {filename}")

        try:
            preprocess_start = time.time()
            try:
                tmp_path, original_size, _ = await asyncio.to_thread(
                    ImageProcessor.preprocess_image, file_content
                )
            except ValueError as e:
                logger.warning(
                    f"[process] | Image Rejected | File: {filename} | {e}"
                )
                raise HTTPException(status_code=400, detail=str(e))

            preprocess_time = round(time.time() - preprocess_start, 3)
            saved_size = os.path.getsize(tmp_path) / 1024
            logger.info(
                f"[process] | [Step 2/3] Image Prepared | File: {filename} | "
                f"Size: {original_size[0]}x{original_size[1]} | "
                f"Saved: {saved_size:.1f} KB | Time: {preprocess_time}s"
            )

            try:
                return await self._process_single(
                    tmp_path,
                    original_size,
                    filename,
                    prompt,
                    start_time,
                )
            except (
                TokenLoopError,
                asyncio.TimeoutError,
                EmptyOutputError,
            ) as first_error:
                logger.warning(
                    f"[process] | Attempt 1 FAILED | File: {filename} | "
                    f"Prompt: '{prompt}' | Error: {type(first_error).__name__}"
                )

                fallback_prompts = []
                if prompt != settings.PROMPT_GENERAL_OCR:
                    fallback_prompts.append(settings.PROMPT_GENERAL_OCR)
                if prompt != settings.PROMPT_FREE_OCR:
                    fallback_prompts.append(settings.PROMPT_FREE_OCR)

                for i, fallback_prompt in enumerate(fallback_prompts, start=2):
                    try:
                        logger.warning(
                            f"[process] | RETRY #{i} | File: {filename} | "
                            f"Trying '{fallback_prompt}'..."
                        )
                        result = await self._process_single(
                            tmp_path,
                            original_size,
                            filename,
                            fallback_prompt,
                            start_time,
                        )
                        result["retried"] = True
                        result["retry_attempt"] = i
                        result["original_prompt_error"] = type(first_error).__name__
                        return result
                    except (
                        TokenLoopError,
                        asyncio.TimeoutError,
                        EmptyOutputError,
                    ) as retry_error:
                        logger.error(
                            f"[process] | Attempt {i} FAILED | File: {filename} | "
                            f"Prompt: '{fallback_prompt}' | "
                            f"Error: {type(retry_error).__name__}: {retry_error}"
                        )
                        continue

                self._raise_http_error(first_error, filename, start_time)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _raise_http_error(error: Exception, filename: str, start_time: float) -> None:
        """Convert exception to appropriate HTTPException."""
        elapsed = round(time.time() - start_time, 3)

        if isinstance(error, asyncio.TimeoutError):
            logger.error(
                f"[process] TIMEOUT | File: {filename} | "
                f"Elapsed: {elapsed}s | Limit: {settings.OLLAMA_TIMEOUT}s"
            )
            raise HTTPException(
                status_code=504,
                detail=(
                    f"OCR timed out after {elapsed}s. "
                    f"Try a smaller image or 'Free OCR.' prompt."
                ),
            )

        if isinstance(error, TokenLoopError):
            logger.warning(
                f"[process] TOKEN LOOP | File: {filename} | "
                f"Elapsed: {elapsed}s | {error}"
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"OCR detected repetitive output after {elapsed}s. "
                    f"Try 'Free OCR.' prompt for this image."
                ),
            )

        if isinstance(error, EmptyOutputError):
            logger.warning(
                f"[process] EMPTY OUTPUT | File: {filename} | "
                f"Elapsed: {elapsed}s | {error}"
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
            f"[process] ERROR | File: {filename} | "
            f"Elapsed: {elapsed}s | {type(error).__name__}: {error}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"OCR failed: {type(error).__name__}: {error}",
        )
