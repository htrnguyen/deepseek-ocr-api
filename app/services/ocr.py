import asyncio
import time
import os
import io
from PIL import Image
import ollama
from fastapi import HTTPException
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BaseService
from app.services.layout import LayoutService
from app.utils.image import ImageProcessor


class OCRService(BaseService):
    def __init__(self):
        self.model = settings.OLLAMA_MODEL
        self.layout = LayoutService()

    async def process(
        self,
        file_content: bytes,
        filename: str,
        prompt: str,
        skip_validation: bool = False,
    ) -> dict:
        start = time.time()
        tmp_path = None

        try:
            tmp_path, original_size, scale = await asyncio.to_thread(
                ImageProcessor.preprocess,
                file_content,
                max_size=None,
                skip_validation=skip_validation,
            )

            layout_result = await self.layout.process(
                file_content, filename, skip_validation=skip_validation
            )
            images = layout_result.get("images", [])

            if not images:
                result = await self._call_ollama(tmp_path, prompt)
                total = round(time.time() - start, 2)
                return self._build_response(
                    {
                        "filename": filename,
                        "text": result["text"],
                        "has_image": False,
                        "processing_time": f"{total}s",
                        "tokens": result["eval_count"],
                    }
                )

            full_img = Image.open(tmp_path)
            text_with_markers = await self._process_with_markers(
                full_img, images, filename, scale, prompt
            )
            total = round(time.time() - start, 2)

            return self._build_response(
                {
                    "filename": filename,
                    "text": text_with_markers,
                    "has_image": True,
                    "image_count": len(images),
                    "processing_time": f"{total}s",
                    "tokens": 0,
                }
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _process_with_markers(
        self, img: Image.Image, images: list, filename: str, scale: float, prompt: str
    ) -> str:
        img_w, img_h = img.size
        image_zones = [(i["bbox"], i) for i in images]
        image_zones.sort(key=lambda x: x[0][1])

        parts = []
        last_y = 0
        img_idx = 0

        for bbox, img_info in image_zones:
            x1, y1, x2, y2 = bbox
            y1_scaled, y2_scaled = int(y1 * scale), int(y2 * scale)

            if y1_scaled > last_y + 50:
                text_region = (0, last_y, img_w, y1_scaled)
                text = await self._ocr_region(img, text_region, filename, prompt)
                if text.strip():
                    parts.append(text)

            img_idx += 1
            parts.append(f"[IMAGE_{img_idx}]")
            last_y = y2_scaled

        if last_y < img_h - 50:
            text_region = (0, last_y, img_w, img_h)
            text = await self._ocr_region(img, text_region, filename, prompt)
            if text.strip():
                parts.append(text)

        return "\n\n".join(parts)

    async def _ocr_region(
        self, img: Image.Image, bbox: tuple, filename: str, prompt: str
    ) -> str:
        def _crop_and_save():
            region = img.crop(bbox)
            tmp_path = f"/tmp/ocr_region_{filename}_{bbox[1]}.jpg"
            region.save(tmp_path, format="JPEG", quality=90)
            return tmp_path

        tmp_path = await asyncio.to_thread(_crop_and_save)
        try:
            with open(tmp_path, "rb") as f:
                region_bytes = f.read()

            region_file = type(
                "File",
                (),
                {"file_content": region_bytes, "filename": f"{filename}_region"},
            )()
            result = await self.process(
                region_bytes, f"{filename}_region", prompt, skip_validation=True
            )
            return result.get("text", "")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _call_ollama(self, tmp_path: str, prompt: str) -> dict:
        def _sync_call():
            tokens = []
            stream = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt, "images": [tmp_path]}],
                options={
                    "temperature": settings.OLLAMA_TEMPERATURE,
                    "num_ctx": settings.OLLAMA_NUM_CTX,
                    "num_predict": settings.OLLAMA_NUM_PREDICT,
                    "repeat_penalty": settings.OLLAMA_REPEAT_PENALTY,
                },
                stream=True,
                keep_alive=-1,
            )

            response_meta = None
            for chunk in stream:
                msg = getattr(chunk, "message", None)
                token = getattr(msg, "content", "") if msg else ""
                if token:
                    tokens.append(token)
                if getattr(chunk, "done", False):
                    response_meta = chunk

            return {
                "text": "".join(tokens),
                "eval_count": getattr(response_meta, "eval_count", len(tokens)),
            }

        return await asyncio.wait_for(
            asyncio.to_thread(_sync_call), timeout=settings.OLLAMA_TIMEOUT
        )

    async def _ocr_region_simple(
        self, img: Image.Image, bbox: tuple, prompt: str
    ) -> str:
        def _crop_and_ocr():
            region = img.crop(bbox)
            buf = io.BytesIO()
            region.save(buf, format="PNG")
            buf.seek(0)

            stream = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt, "images": [buf.getvalue()]}
                ],
                options={
                    "temperature": settings.OLLAMA_TEMPERATURE,
                    "num_ctx": settings.OLLAMA_NUM_CTX,
                    "num_predict": settings.OLLAMA_NUM_PREDICT,
                },
                stream=True,
                keep_alive=-1,
            )

            tokens = []
            for chunk in stream:
                msg = getattr(chunk, "message", None)
                token = getattr(msg, "content", "") if msg else ""
                if token:
                    tokens.append(token)

            return "".join(tokens)

        return await asyncio.to_thread(_crop_and_ocr)
