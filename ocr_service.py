import asyncio
import time
import os
import tempfile
from pathlib import Path
from PIL import Image
from io import BytesIO
import ollama
from config import settings
from image_utils import enhance_for_ocr, auto_resize_image
import logging

logger = logging.getLogger("deepseek-ocr-api")


class DeepSeekOCRService:
    def __init__(self):
        self.model = settings.OLLAMA_MODEL

    async def process(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
    ):
        start_time = time.time()
        tmp_path = None

        logger.info(f"[OCR START] File: {filename} | Prompt: '{prompt}'")

        try:
            # --- Image preprocessing ---
            preprocess_start = time.time()
            with Image.open(BytesIO(file_content)) as img:
                original_size = img.size
                logger.info(
                    f"[IMAGE] Original: {original_size[0]}x{original_size[1]} | "
                    f"Mode: {img.mode} | File size: {len(file_content) / 1024:.1f} KB"
                )

                if img.mode != "RGB":
                    img = img.convert("RGB")

                img = enhance_for_ocr(img)
                img = auto_resize_image(img, settings.MAX_LONG_SIDE)

                processed_size = img.size
                suffix = Path(filename).suffix.lower()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_path = tmp.name
                img.save(tmp_path, quality=98 if suffix in [".jpg", ".jpeg"] else None)

            preprocess_time = round(time.time() - preprocess_start, 3)
            logger.info(
                f"[IMAGE] Processed: {processed_size[0]}x{processed_size[1]} | "
                f"Preprocessing: {preprocess_time}s"
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
                f"[OLLAMA CONFIG] Model: {self.model} | "
                f"Timeout: {settings.OLLAMA_TIMEOUT}s | "
                f"num_ctx: {ollama_options['num_ctx']} | "
                f"num_predict: {ollama_options['num_predict']} | "
                f"temperature: {ollama_options['temperature']} | "
                f"repeat_penalty: {ollama_options['repeat_penalty']} | "
                f"repeat_last_n: {ollama_options['repeat_last_n']} | "
                f"top_k: {ollama_options['top_k']} | "
                f"top_p: {ollama_options['top_p']}"
            )

            # --- Call Ollama ---
            ollama_start = time.time()
            logger.info("[OLLAMA] Sending request to Ollama...")

            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.chat,
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [tmp_path],
                        }
                    ],
                    options=ollama_options,
                    keep_alive=600,
                ),
                timeout=settings.OLLAMA_TIMEOUT,
            )

            ollama_time = round(time.time() - ollama_start, 3)

            # --- Extract response ---
            result_text = response.get("message", {}).get("content", "").strip()
            prompt_tokens = response.get("prompt_eval_count", 0) or 0
            response_tokens = response.get("eval_count", 0) or 0
            total_tokens = prompt_tokens + response_tokens
            total_time = round(time.time() - start_time, 3)

            # Token speed (tokens/sec)
            token_speed = round(response_tokens / ollama_time, 1) if ollama_time > 0 else 0

            logger.info(
                f"[OCR DONE] Total: {total_time}s | Ollama: {ollama_time}s | "
                f"Tokens: {prompt_tokens} prompt + {response_tokens} response = {total_tokens} total | "
                f"Speed: {token_speed} tok/s | "
                f"Output length: {len(result_text)} chars"
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
                f"[OCR TIMEOUT] File: {filename} | "
                f"Elapsed: {elapsed}s | Limit: {settings.OLLAMA_TIMEOUT}s | "
                f"Possible cause: token loop or large image"
            )
            raise

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            logger.error(
                f"[OCR ERROR] File: {filename} | "
                f"Elapsed: {elapsed}s | Error: {type(e).__name__}: {e}"
            )
            raise

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
