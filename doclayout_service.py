import time
import os
import tempfile
import asyncio
from io import BytesIO
from pathlib import Path

from fastapi import HTTPException
from PIL import Image, ImageOps
from doclayout_yolo import YOLOv10

from config import settings
from logger import logger


class DocLayoutService:
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info("[detect_figures] Lazy-loading DocLayout-YOLO model...")
            self.model = YOLOv10(settings.DOC_LAYOUT_MODEL_PATH)
            logger.info("[detect_figures] DocLayout-YOLO model loaded successfully")

    async def detect_figures(self, file_content: bytes, filename: str):
        # Lazy load model in background thread to avoid blocking event loop
        if self.model is None:
            await asyncio.to_thread(self._load_model)

        start_time = time.time()
        tmp_path = None

        try:
            suffix = Path(filename).suffix.lower()
            if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
                suffix = ".jpg"

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_path = tmp.name

            # Load with PIL to fix EXIF rotation
            with Image.open(BytesIO(file_content)) as img:
                img = ImageOps.exif_transpose(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(tmp_path, format="JPEG", quality=95)

            tmp.close()

            logger.info(f"[detect_figures] DocLayout-YOLO figure detection started - File: {filename}")

            results = await asyncio.to_thread(
                self.model.predict, tmp_path, imgsz=1024, conf=0.25, verbose=False
            )
            image_regions = []

            if len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    figure_mask = (result.boxes.cls == 3).squeeze()
                    if figure_mask.any():
                        figure_boxes = result.boxes.xyxy[figure_mask].cpu().numpy()
                        figure_conf = result.boxes.conf[figure_mask].cpu().numpy()
                        for box, conf in zip(figure_boxes, figure_conf):
                            x1, y1, x2, y2 = map(int, box)
                            image_regions.append(
                                {
                                    "label": "image",
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": round(float(conf), 4),
                                    "width": x2 - x1,
                                    "height": y2 - y1,
                                }
                            )

            process_time = round(time.time() - start_time, 3)
            logger.info(
                f"[detect_figures] DocLayout-YOLO completed in {process_time}s - Found {len(image_regions)} figure(s)"
            )

            return {
                "image_count": len(image_regions),
                "images": image_regions,
                "processing_time": f"{process_time}s",
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
