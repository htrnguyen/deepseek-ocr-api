import asyncio
import re
import time
import os
import tempfile
from PIL import Image, ImageOps
from io import BytesIO
import ollama
from fastapi import HTTPException
from config import settings
from logger import logger


class TokenLoopError(Exception):
    """Raised when repetitive token output is detected during streaming."""

    pass


class EmptyOutputError(Exception):
    """Raised when OCR output is suspiciously short (likely model hallucination)."""

    pass


class DeepSeekOCRService:
    """OCR service using DeepSeek model via Ollama with streaming monitoring.

    DeepSeek-OCR uses Dynamic Resolution internally:
    - Global view: 1×1024×1024 → 256 visual tokens
    - Local patches: (1-6)×768×768 → (1-6)×144 visual tokens each
    → We send the ORIGINAL image (EXIF-fixed only) and let the model handle patching.

    Key features:
    - NO image preprocessing (model handles resolution internally)
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
        Strategy 3: Character uniqueness + Substring repetition
        Strategy 4: Regex patterns — known HTML/Markdown loop patterns
        """
        if len(tokens) < 20:
            return False

        # Strategy 1: Unique token ratio (MOST IMPORTANT)
        window = tokens[-100:] if len(tokens) >= 100 else tokens
        if len(window) >= 40:
            unique_ratio = len(set(window)) / len(window)
            if unique_ratio < 0.25:
                logger.warning(
                    f"[loop_detect] Strategy 1 (Unique Ratio): {unique_ratio:.2%} < 25%. "
                    f"Tokens: {len(set(window))} unique / {len(window)} total. "
                    f"Sample: {''.join(window[-10:])}"
                )
                return True

        # Strategy 2: Window comparison
        if len(tokens) >= 60:
            recent = "".join(tokens[-30:])
            earlier = "".join(tokens[-60:-30])
            if recent == earlier and len(recent) > 10:
                logger.warning(
                    f"[loop_detect] Strategy 2 (Window Match): "
                    f"30-token window repeated exactly. Match: '{recent[:30]}...'"
                )
                return True

        # Strategy 3: Character uniqueness and Exact Substring Repeat
        recent_text = "".join(tokens[-150:]) if len(tokens) >= 150 else "".join(tokens)

        # 3a. Character uniqueness
        if len(recent_text) >= 50:
            unique_chars = len(set(recent_text))
            if unique_chars < 15:
                logger.warning(
                    f"[loop_detect] Strategy 3a (Char Uniqueness): "
                    f"{unique_chars} unique chars in {len(recent_text)} length text. "
                    f"Sample: '{recent_text[-30:]}'"
                )
                return True

        # 3b. Long Substring Repetition
        if len(recent_text) >= 60:
            for p_len in range(15, 40):
                pattern = recent_text[-p_len:]
                if recent_text.endswith(pattern * 3):
                    logger.warning(
                        f"[loop_detect] Strategy 3b (Substring Repeat): "
                        f"Pattern '{pattern}' (len {p_len}) repeated 3+ times at end."
                    )
                    return True

        # Strategy 4: Known HTML/Markdown patterns
        for pattern in self.LOOP_PATTERNS:
            if pattern.search(recent_text):
                logger.warning(
                    f"[loop_detect] Strategy 4 (Regex Pattern): "
                    f"Matched pattern {pattern.pattern} on text: '{recent_text[-40:]}'"
                )
                return True

        return False

    @staticmethod
    def _clean_output(text: str) -> str:
        """Post-process OCR output to remove grounding artifacts based on official DeepSeek-OCR v2 logic.

        Grounding mode outputs: <|ref|>word<|/ref|><|det|>[[x,y,x,y]]<|/det|>
        """
        if not text:
            return ""

        # Remove <|ref|>...<|/ref|><|det|>...<|/det|> entirely if it references an 'image'
        # Otherwise, remove the grounding line
        pattern = r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)"
        matches = re.findall(pattern, text, re.DOTALL)

        for match in matches:
            if "<|ref|>image<|/ref|>" in match[0]:
                text = text.replace(match[0], "", 1)
            else:
                # Remove the exact matched line
                text = re.sub(rf"(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?", "", text)

        # Replace mathematical symbols
        text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

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
        try:
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

                # Progress log every 50 tokens
                if len(tokens) > 0 and len(tokens) % 50 == 0:
                    elapsed = round(time.time() - stream_start, 1)
                    preview = "".join(tokens[-20:])[:80]
                    logger.info(
                        f"[streaming] Progress: {len(tokens)} tokens | "
                        f"{elapsed}s | Last: '{preview}'"
                    )

                # Check for token loop every 10 tokens (aggressive)
                if len(tokens) >= 20 and len(tokens) % 10 == 0:
                    if self._detect_token_loop(tokens):
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
                        raise TokenLoopError(
                            f"Loop after {len(tokens)} tokens ({elapsed}s). "
                            f"Last: '{''.join(tokens[-8:])[:60]}...'"
                        )

                if chunk.get("done"):
                    response_meta = chunk

        except ollama.ResponseError as e:
            # Ollama's built-in loop detection throws this error
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
                raise TokenLoopError(
                    f"Ollama loop detection after {len(tokens)} tokens ({elapsed}s)"
                )
            raise  # Re-raise non-loop errors

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
            f"[process] Ollama: model={self.model} | prompt='{prompt}' | "
            f"timeout={settings.OLLAMA_TIMEOUT}s | "
            f"repeat_penalty={ollama_options['repeat_penalty']}"
        )

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
            f"[process] ✅ OCR done | Total: {total_time}s | Ollama: {ollama_time}s | "
            f"Tokens: {prompt_tokens}p + {response_tokens}r = {total_tokens} | "
            f"Speed: {token_speed} tok/s | Output: {len(result_text)} chars\n"
            f"  Preview: '{result_text[:200]}'"
        )

        # Guard: if output is suspiciously short, it's likely hallucination
        if not result_text or (len(result_text) < 20 and response_tokens < 50):
            logger.warning(
                f"[process] EmptyOutput: only {len(result_text)} chars / "
                f"{response_tokens} tokens — likely hallucination"
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
        """Process an image through DeepSeek OCR.

        Pipeline: Open → EXIF Transpose → RGB → Save JPEG → OCR
        NO resize/enhance — DeepSeek handles Dynamic Resolution internally
        (Global 1024×1024 + Local patches 768×768, 1-6 patches).

        Auto-retry: if grounding mode fails, fallback to "Free OCR." prompt.
        """
        start_time = time.time()
        tmp_path = None

        logger.info(f"[process] OCR start | File: {filename}")

        try:
            preprocess_start = time.time()
            with Image.open(BytesIO(file_content)) as img:
                # Step 1: Fix EXIF rotation (smartphone photos)
                img = ImageOps.exif_transpose(img)

                original_size = img.size
                logger.info(
                    f"[process] Image: {original_size[0]}x{original_size[1]} | "
                    f"Mode: {img.mode} | Size: {len(file_content) / 1024:.1f} KB"
                )

                # Step 2: Convert to RGB (required for JPEG save)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Step 3: Save as JPEG (Ollama needs a file path)
                # No resize, no enhance — let DeepSeek's Dynamic Resolution handle it
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    tmp_path = tmp.name
                    img.save(tmp_path, format="JPEG", quality=95)

            preprocess_time = round(time.time() - preprocess_start, 3)
            saved_size = os.path.getsize(tmp_path) / 1024
            logger.info(
                f"[process] Prepared: {original_size[0]}x{original_size[1]} | "
                f"Saved: {saved_size:.1f} KB | Time: {preprocess_time}s"
            )

            # --- Attempt 1: Original prompt ---
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
                    f"[process] ⚠️ Attempt 1 FAILED | prompt='{prompt}' | "
                    f"{type(first_error).__name__}"
                )

                # Build fallback prompt chain (skip current prompt)
                fallback_prompts = []
                if prompt != settings.PROMPT_GENERAL_OCR:
                    fallback_prompts.append(settings.PROMPT_GENERAL_OCR)
                if prompt != settings.PROMPT_FREE_OCR:
                    fallback_prompts.append(settings.PROMPT_FREE_OCR)

                # --- Attempt 2 & 3: Fallback prompts ---
                for i, fallback_prompt in enumerate(fallback_prompts, start=2):
                    try:
                        logger.warning(
                            f"[process] RETRY #{i} | Trying '{fallback_prompt}'..."
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
                            f"[process] ❌ Attempt {i} FAILED | "
                            f"prompt='{fallback_prompt}' | "
                            f"{type(retry_error).__name__}: {retry_error}"
                        )
                        continue  # Try next fallback

                # All attempts failed
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
