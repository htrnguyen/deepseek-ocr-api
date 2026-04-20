import asyncio
import re
import time
import os
import tempfile
from pathlib import Path
from PIL import Image
from io import BytesIO
import ollama
from fastapi import HTTPException
from config import settings
from image_utils import (
    enhance_for_ocr,
    auto_resize_image,
    calculate_save_quality,
    smart_crop_content_region,
)
from logger import logger


class TokenLoopError(Exception):
    """Raised when repetitive token output is detected during streaming."""

    pass


class DeepSeekOCRService:
    """OCR service using DeepSeek model via Ollama with streaming monitoring.

    Key features:
    - Streaming inference with aggressive token loop detection (every 10 tokens)
    - Auto-retry: grounding mode fails → fallback to "Free OCR."
    - Post-processing: clean grounding tags from output
    - keep_alive=-1: model stays loaded permanently
    """

    # Patterns that indicate a token loop
    LOOP_PATTERNS = [
        re.compile(r"(<td>\s*</td>){6,}"),  # Empty table cells
        re.compile(r"(<tr>\s*</tr>){4,}"),  # Empty table rows
        re.compile(r"(\|\s*){12,}"),  # Markdown table pipes
        re.compile(r"(</td><td>){6,}"),  # Alternating td tags
        re.compile(r"(<td></td>){5,}"),  # Compact empty cells
    ]

    def __init__(self, doclayout_model=None):
        self.model = settings.OLLAMA_MODEL
        self.doclayout_model = doclayout_model

    def _detect_token_loop(self, tokens: list[str]) -> bool:
        """Detect repetitive output using 4 strategies.

        Strategy 1: Unique token ratio — catches ANY repetition pattern
        Strategy 2: Window comparison — exact match of consecutive windows
        Strategy 3: Character uniqueness — very few unique chars = loop
        Strategy 4: Regex patterns — known HTML/Markdown loop patterns
        """
        if len(tokens) < 20:
            return False

        # Strategy 1: Unique token ratio (MOST IMPORTANT)
        # Increase window to 100 to handle multi-token characters (like Chinese)
        window = tokens[-100:] if len(tokens) >= 100 else tokens
        if len(window) >= 40:
            unique_ratio = len(set(window)) / len(window)
            if unique_ratio < 0.25:  # Relaxed from 0.2
                logger.warning(
                    f"[loop_detect] Strategy 1 (Unique Ratio): {unique_ratio:.2%} < 25%. Tokens: {len(set(window))} unique / {len(window)} total. Sample: {''.join(window[-10:])}"
                )
                return True

        # Strategy 2: Window comparison
        if len(tokens) >= 60:
            recent = "".join(tokens[-30:])
            earlier = "".join(tokens[-60:-30])
            if recent == earlier and len(recent) > 10:
                logger.warning(
                    f"[loop_detect] Strategy 2 (Window Match): 30-token window repeated exactly. Match: '{recent[:30]}...'"
                )
                return True

        # Strategy 3: Character uniqueness and Exact Substring Repeat
        recent_text = "".join(tokens[-150:]) if len(tokens) >= 150 else "".join(tokens)

        # 3a. Character uniqueness
        if len(recent_text) >= 50:
            unique_chars = len(set(recent_text))
            if unique_chars < 15:
                logger.warning(
                    f"[loop_detect] Strategy 3a (Char Uniqueness): {unique_chars} unique chars in {len(recent_text)} length text. Sample: '{recent_text[-30:]}'"
                )
                return True

        # 3b. Long Substring Repetition (e.g. "如果某市的市名是“市” | " repeating)
        # If a sequence of >10 chars repeats 3+ times in the recent text, it's a loop.
        if len(recent_text) >= 60:
            # Check for patterns of length 15 to 40 characters
            for p_len in range(15, 40):
                pattern = recent_text[-p_len:]
                # If the exact pattern appears 3 consecutive times at the end
                if recent_text.endswith(pattern * 3):
                    logger.warning(
                        f"[loop_detect] Strategy 3b (Substring Repeat): Pattern '{pattern}' (len {p_len}) repeated 3+ times at end."
                    )
                    return True

        # Strategy 4: Known HTML/Markdown patterns
        for pattern in self.LOOP_PATTERNS:
            if pattern.search(recent_text):
                logger.warning(
                    f"[loop_detect] Strategy 4 (Regex Pattern): Matched pattern {pattern.pattern} on text: '{recent_text[-40:]}'"
                )
                return True

        return False

    @staticmethod
    def _clean_output(text: str) -> str:
        """Post-process OCR output to remove grounding artifacts."""
        # Remove <|ref|>text<|/ref|> → keep text
        text = re.sub(r"<\|ref\|>(.*?)<\|/ref\|>", r"\1", text)
        # Remove <|det|>[[...]]<|/det|> → remove entirely
        text = re.sub(r"<\|det\|>\[\[.*?\]\]<\|/det\|>", "", text)
        # Clean excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _call_ollama_streaming(
        self,
        tmp_path: str,
        prompt: str,
        ollama_options: dict,
    ) -> dict:
        """Blocking call to Ollama with streaming and aggressive loop monitoring.

        Checks every 10 tokens. Raises TokenLoopError on detection.
        """
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
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                tokens.append(token)

                # Log first token timing (prompt eval duration)
                if first_token_time is None:
                    first_token_time = time.time()
                    eval_time = round(first_token_time - stream_start, 1)
                    logger.info(
                        f"[streaming] First token after {eval_time}s (prompt eval)"
                    )

            # Check for token loop every 10 tokens (aggressive)
            if len(tokens) >= 20 and len(tokens) % 10 == 0:
                if self._detect_token_loop(tokens):
                    elapsed = round(time.time() - stream_start, 1)
                    sample = "".join(tokens[-8:])[:60]
                    raise TokenLoopError(
                        f"Loop after {len(tokens)} tokens ({elapsed}s). "
                        f"Last: '{sample}...'"
                    )

            if chunk.get("done"):
                response_meta = chunk

        full_text = "".join(tokens)

        # Post-generation loop check: if hit num_predict exactly, likely a loop
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
        processed_size: tuple[int, int],
        filename: str,
        prompt: str,
        start_time: float,
    ) -> dict:
        """Single OCR attempt with given prompt."""
        # Build Ollama options
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
            f"[process] Ollama: model={self.model} | prompt='{prompt}' | "
            f"timeout={settings.OLLAMA_TIMEOUT}s | "
            f"repeat_penalty={ollama_options['repeat_penalty']}"
        )

        # Call Ollama with streaming
        ollama_start = time.time()
        logger.info("[process] Sending streaming request to Ollama...")

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
        result_text = self._clean_output(result["text"]).strip()
        prompt_tokens = result["prompt_eval_count"]
        response_tokens = result["eval_count"]
        total_tokens = prompt_tokens + response_tokens
        total_time = round(time.time() - start_time, 3)
        token_speed = round(response_tokens / ollama_time, 1) if ollama_time > 0 else 0

        logger.info(
            f"[process] OCR done | Total: {total_time}s | Ollama: {ollama_time}s | "
            f"Tokens: {prompt_tokens}p + {response_tokens}r = {total_tokens} | "
            f"Speed: {token_speed} tok/s | Output: {len(result_text)} chars"
        )

        return {
            "text": result_text,
            "original_size": f"{original_size[0]}x{original_size[1]}",
            "processed_size": f"{processed_size[0]}x{processed_size[1]}",
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
        """Process an image through DeepSeek OCR.

        Auto-retry: if grounding mode fails (loop/timeout/error),
        automatically retries with "Free OCR." prompt.
        """
        start_time = time.time()
        tmp_path = None

        logger.info(f"[process] OCR start | File: {filename}")

        try:
            # --- Image preprocessing ---
            # Pipeline: Open → EXIF Transpose → RGB → Resize → Enhance → Smart Crop → Save
            preprocess_start = time.time()
            from PIL import ImageOps
            with Image.open(BytesIO(file_content)) as img:
                # Apply EXIF rotation automatically (crucial for smartphone photos)
                img = ImageOps.exif_transpose(img)
                
                original_size = img.size
                logger.info(
                    f"[process] Image: {original_size[0]}x{original_size[1]} | "
                    f"Mode: {img.mode} | Size: {len(file_content) / 1024:.1f} KB"
                )

                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Step 1: Resize (critical for 4K+ images, preserves quality via LANCZOS)
                img = auto_resize_image(img, settings.MAX_LONG_SIDE)

                # Step 2: Enhance (Contrast + Sharpness)
                img = enhance_for_ocr(img, original_resolution=original_size)

                # Bỏ qua Smart Crop theo yêu cầu để lấy toàn bộ trang giấy
                crop_info = {"cropped": False, "skip_reason": "Disabled by user"}

                processed_size = img.size
                save_quality = calculate_save_quality(len(file_content), processed_size)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp_path = tmp.name
                    img.save(tmp_path, format="JPEG", quality=save_quality)

            preprocess_time = round(time.time() - preprocess_start, 3)
            saved_size = os.path.getsize(tmp_path) / 1024
            logger.info(
                f"[process] Preprocessed: {processed_size[0]}x{processed_size[1]} | "
                f"Quality: {save_quality} | Saved: {saved_size:.1f} KB | "
                f"Time: {preprocess_time}s"
            )

            # --- Attempt 1: Original prompt ---
            try:
                return await self._process_single(
                    tmp_path,
                    original_size,
                    processed_size,
                    filename,
                    prompt,
                    start_time,
                )
            except (TokenLoopError, asyncio.TimeoutError, Exception) as first_error:
                # --- Attempt 2: Auto-retry with Free OCR if grounding failed ---
                is_grounding = "<|grounding|>" in prompt
                is_retryable = isinstance(
                    first_error, (TokenLoopError, asyncio.TimeoutError)
                ) or "RemoteProtocolError" in str(type(first_error).__name__)

                if is_grounding and is_retryable:
                    logger.warning(
                        f"[process] RETRY | '{prompt}' failed "
                        f"({type(first_error).__name__}). "
                        f"Retrying with 'Free OCR.'..."
                    )
                    try:
                        result = await self._process_single(
                            tmp_path,
                            original_size,
                            processed_size,
                            filename,
                            "Free OCR.",
                            start_time,
                        )
                        result["retried"] = True
                        result["original_prompt_error"] = (
                            f"{type(first_error).__name__}"
                        )
                        return result
                    except Exception as retry_error:
                        logger.error(
                            f"[process] RETRY ALSO FAILED | "
                            f"{type(retry_error).__name__}: {retry_error}"
                        )
                        # Fall through to raise original error

                # Re-raise with proper HTTP status
                self._raise_http_error(first_error, filename, start_time)

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @staticmethod
    def _raise_http_error(error: Exception, filename: str, start_time: float):
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
