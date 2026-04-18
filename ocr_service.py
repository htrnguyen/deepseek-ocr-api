import asyncio
import time
import os
import tempfile
from pathlib import Path
from PIL import Image
from io import BytesIO
from fastapi import HTTPException
import ollama
from config import settings
from image_utils import enhance_for_ocr, auto_resize_image
from logger import logger


class DeepSeekOCRService:
    def __init__(self):
        self.model = settings.OLLAMA_MODEL

    async def process(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        temperature: float = 0.0,
        num_ctx: int = 12288,
        num_predict: int = -1,
    ):
        start_time = time.time()
        tmp_path = None

        logger.info(f"START DeepSeek OCR - File: {filename}")

        try:
            with Image.open(BytesIO(file_content)) as img:
                original_size = img.size
                logger.info(
                    f"Original image size: {original_size[0]}x{original_size[1]}"
                )

                if img.mode != "RGB":
                    img = img.convert("RGB")

                img = enhance_for_ocr(img)
                img = auto_resize_image(img, settings.MAX_LONG_SIDE)

                processed_size = img.size
                logger.info(
                    f"Processed image size: {processed_size[0]}x{processed_size[1]}"
                )

                suffix = Path(filename).suffix.lower()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp_path = tmp.name
                img.save(tmp_path, quality=98 if suffix in [".jpg", ".jpeg"] else None)

            # Call Ollama
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    ollama.chat,
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt, "images": [tmp_path]}
                    ],
                    options={
                        "temperature": temperature,
                        "num_ctx": num_ctx,
                        "num_predict": num_predict,
                    },
                    keep_alive=180,
                ),
                timeout=settings.OLLAMA_TIMEOUT,
            )

            result_text = response.get("message", {}).get("content", "").strip()
            total_time = round(time.time() - start_time, 3)

            logger.info(
                f"DeepSeek OCR completed in {total_time}s | Tokens: {response.get('prompt_eval_count', 0)} prompt / {response.get('eval_count', 0)} response"
            )

            return {
                "text": result_text,
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "processed_size": f"{processed_size[0]}x{processed_size[1]}",
                "processing_time": f"{total_time}s",
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "response_tokens": response.get("eval_count", 0),
                "total_tokens": (response.get("prompt_eval_count") or 0)
                + (response.get("eval_count") or 0),
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
