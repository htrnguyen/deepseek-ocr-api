import asyncio
import time
import os
from PIL import Image
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BaseService
from app.services.ocr import OCRService
from app.services.layout import LayoutService
from app.utils.image import ImageProcessor


class OcrLayoutService(BaseService):
    def __init__(self):
        self.ocr = OCRService()
        self.layout = LayoutService()

    async def process(self, file_content: bytes, filename: str) -> dict:
        start = time.time()
        tmp_path = None

        try:
            tmp_path, original_size, scale = await asyncio.to_thread(
                ImageProcessor.preprocess, file_content, max_size=1536, skip_validation=True
            )

            layout_result = await self.layout.process(file_content, filename)
            images = layout_result.get("images", [])

            if not images:
                ocr_result = await self.ocr.process(file_content, filename, settings.DEFAULT_PROMPT)
                return self._build_response({
                    "filename": filename,
                    "markdown": ocr_result["text"],
                    "blocks": [{"type": "text", "content": ocr_result["text"]}],
                    "has_images": False,
                })

            full_img = Image.open(tmp_path)
            blocks = await self._process_blocks(full_img, images, filename, scale)
            markdown = self._build_markdown(blocks)

            elapsed = round(time.time() - start, 2)

            return self._build_response({
                "filename": filename,
                "markdown": markdown,
                "blocks": blocks,
                "image_count": len(images),
                "has_images": True,
                "processing_time": f"{elapsed}s",
            })

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _process_blocks(self, img: Image.Image, images: list, filename: str, scale: float) -> list:
        img_w, img_h = img.size
        image_zones = [(i["bbox"], i) for i in images]
        image_zones.sort(key=lambda x: x[0][1])

        blocks = []
        last_y = 0

        for bbox, img_info in image_zones:
            x1, y1, x2, y2 = bbox
            y1_scaled, y2_scaled = int(y1 * scale), int(y2 * scale)

            if y1_scaled > last_y + 50:
                text_region = (0, last_y, img_w, y1_scaled)
                text = await self._ocr_region(img, text_region, filename)
                if text.strip():
                    blocks.append({"type": "text", "content": text, "bbox": text_region})

            blocks.append({
                "type": "image",
                "bbox": bbox,
                "confidence": img_info.get("confidence", 0),
            })

            last_y = y2_scaled

        if last_y < img_h - 50:
            text_region = (0, last_y, img_w, img_h)
            text = await self._ocr_region(img, text_region, filename)
            if text.strip():
                blocks.append({"type": "text", "content": text, "bbox": text_region})

        return blocks

    async def _ocr_region(self, img: Image.Image, bbox: tuple, filename: str) -> str:
        def _crop_and_save():
            region = img.crop(bbox)
            tmp_path = f"/tmp/ocr_region_{filename}_{bbox[1]}.jpg"
            region.save(tmp_path, format="JPEG", quality=90)
            return tmp_path

        tmp_path = await asyncio.to_thread(_crop_and_save)
        try:
            with open(tmp_path, "rb") as f:
                region_bytes = f.read()
            result = await self.ocr.process(region_bytes, f"{filename}_region", settings.DEFAULT_PROMPT)
            return result.get("text", "")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _build_markdown(self, blocks: list) -> str:
        parts = []
        for i, block in enumerate(blocks):
            if block["type"] == "text":
                parts.append(block["content"])
            elif block["type"] == "image":
                parts.append(f"\n[IMAGE_{i}]\n")
        return "\n\n".join(parts)
