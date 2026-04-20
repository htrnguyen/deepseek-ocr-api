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
from image_utils import enhance_for_ocr, auto_resize_image, calculate_save_quality
from logger import logger


class TokenLoopError(Exception):
    """Raised when repetitive token output is detected during streaming."""
    pass


class DeepSeekOCRService:
    """OCR service using DeepSeek model via Ollama with streaming monitoring.

    Key features:
    - Streaming inference with real-time token loop detection
    - Adaptive image preprocessing (resize-first, skip sharpening on hi-res)
    - keep_alive=-1: model stays loaded permanently
    - Post-processing: clean grounding tags and malformed output
    """

    # Patterns that indicate a token loop (repetitive structural tags)
    LOOP_PATTERNS = [
        re.compile(r"(<td>\s*</td>){10,}"),           # Empty table cells
        re.compile(r"(<tr>\s*</tr>){5,}"),             # Empty table rows
        re.compile(r"(\|\s*){20,}"),                   # Markdown table pipes
        re.compile(r"(</td><td>){10,}"),               # Alternating td tags
        re.compile(r"(.{3,}?)\1{5,}"),                 # Any 3+ char pattern repeated 5+ times
    ]

    def __init__(self):
        self.model = settings.OLLAMA_MODEL

    def _detect_token_loop(self, tokens: list[str]) -> bool:
        """Detect repetitive output using multiple strategies.

        Strategy 1: Window comparison — last 20 tokens vs previous 20
        Strategy 2: Pattern matching — known loop patterns (td, tr, pipes)
        """
        if len(tokens) < 30:
            return False

        # Strategy 1: Exact window match
        recent = "".join(tokens[-15:])
        earlier = "".join(tokens[-30:-15])
        if recent == earlier and len(recent) > 10:
            return True

        # Strategy 2: Check known loop patterns in recent output
        recent_text = "".join(tokens[-40:]) if len(tokens) >= 40 else "".join(tokens)
        for pattern in self.LOOP_PATTERNS:
            if pattern.search(recent_text):
                return True

        return False

    @staticmethod
    def _clean_output(text: str) -> str:
        """Post-process OCR output to remove grounding artifacts.

        DeepSeek OCR in grounding mode outputs bounding box tags like:
        <|ref|>text<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>
        These are useful for localization but not for text extraction.
        """
        # Remove grounding reference tags: <|ref|>...<|/ref|>
        text = re.sub(r"<\|ref\|>(.*?)<\|/ref\|>", r"\1", text)
        # Remove detection bounding boxes: <|det|>[[...]]<|/det|>
        text = re.sub(r"<\|det\|>\[\[.*?\]\]<\|/det\|>", "", text)
        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _call_ollama_streaming(
        self,
        tmp_path: str,
        prompt: str,
        ollama_options: dict,
    ) -> dict:
        """Blocking call to Ollama with streaming and token loop monitoring.

        Monitors every 20 tokens for:
        - Token loop (window comparison + pattern matching)
        - num_predict cap (Ollama auto-stops)

        Returns dict with text, prompt_eval_count, eval_count.
        """
        tokens: list[str] = []
        stream_start = time.time()

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

            # Check for token loop every 20 tokens (aggressive detection)
            if len(tokens) % 20 == 0 and self._detect_token_loop(tokens):
                elapsed = round(time.time() - stream_start, 1)
                token_count = len(tokens)
                sample = "".join(tokens[-10:])[:80]
                raise TokenLoopError(
                    f"Repetitive output after {token_count} tokens ({elapsed}s). "
                    f"Sample: '{sample}...'"
                )

            if chunk.get("done"):
                response_meta = chunk

        return {
            "text": "".join(tokens),
            "prompt_eval_count": response_meta.get("prompt_eval_count", 0) or 0,
            "eval_count": response_meta.get("eval_count", 0) or 0,
        }

    async def process(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
    ) -> dict:
        """Process an image through DeepSeek OCR with streaming and monitoring."""
        start_time = time.time()
        tmp_path = None

        logger.info(f"[process] OCR start | File: {filename} | Prompt: '{prompt}'")

        try:
            # --- Image preprocessing ---
            # Pipeline: Open → RGB → Resize → Enhance → Save
            # IMPORTANT: Resize BEFORE enhance to avoid processing 4K pixels
            preprocess_start = time.time()
            with Image.open(BytesIO(file_content)) as img:
                original_size = img.size
                logger.info(
                    f"[process] Image: {original_size[0]}x{original_size[1]} | "
                    f"Mode: {img.mode} | Size: {len(file_content) / 1024:.1f} KB"
                )

                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Step 1: Resize first (critical for 4K+ images)
                img = auto_resize_image(img, settings.MAX_LONG_SIDE)

                # Step 2: Enhance on resized image (fast + adaptive)
                img = enhance_for_ocr(img, original_resolution=original_size)

                processed_size = img.size
                save_quality = calculate_save_quality(
                    len(file_content), processed_size
                )

                # Always save as JPEG for consistent, compact temp files
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as tmp:
                    tmp_path = tmp.name
                    img.save(tmp_path, format="JPEG", quality=save_quality)

            preprocess_time = round(time.time() - preprocess_start, 3)
            saved_size = os.path.getsize(tmp_path) / 1024
            logger.info(
                f"[process] Preprocessed: {processed_size[0]}x{processed_size[1]} | "
                f"Quality: {save_quality} | Saved: {saved_size:.1f} KB | "
                f"Time: {preprocess_time}s"
            )

            # --- Build Ollama options ---
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
                f"[process] Ollama: model={self.model} | "
                f"timeout={settings.OLLAMA_TIMEOUT}s | "
                f"repeat_penalty={ollama_options['repeat_penalty']} | "
                f"repeat_last_n={ollama_options['repeat_last_n']}"
            )

            # --- Call Ollama with streaming ---
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

            # --- Post-process & extract response ---
            raw_text = result["text"]
            result_text = self._clean_output(raw_text).strip()
            prompt_tokens = result["prompt_eval_count"]
            response_tokens = result["eval_count"]
            total_tokens = prompt_tokens + response_tokens
            total_time = round(time.time() - start_time, 3)
            token_speed = (
                round(response_tokens / ollama_time, 1) if ollama_time > 0 else 0
            )

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

        except asyncio.TimeoutError:
            elapsed = round(time.time() - start_time, 3)
            logger.error(
                f"[process] TIMEOUT | File: {filename} | "
                f"Elapsed: {elapsed}s | Limit: {settings.OLLAMA_TIMEOUT}s"
            )
            raise HTTPException(
                status_code=504,
                detail=(
                    f"OCR timed out after {elapsed}s "
                    f"(limit: {settings.OLLAMA_TIMEOUT}s). Please retry."
                ),
            )

        except TokenLoopError as e:
            elapsed = round(time.time() - start_time, 3)
            logger.warning(
                f"[process] TOKEN LOOP | File: {filename} | "
                f"Elapsed: {elapsed}s | {e}"
            )
            raise HTTPException(
                status_code=422,
                detail=(
                    f"OCR detected repetitive output (token loop) after {elapsed}s. "
                    f"Try using 'Free OCR.' prompt instead of grounding mode."
                ),
            )

        except HTTPException:
            raise

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            logger.error(
                f"[process] ERROR | File: {filename} | "
                f"Elapsed: {elapsed}s | {type(e).__name__}: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"OCR failed: {type(e).__name__}: {e}",
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
