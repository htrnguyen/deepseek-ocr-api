import asyncio
import time
import os
import threading
from doclayout_yolo import YOLOv10
from fastapi import HTTPException
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BaseService
from app.utils.image import ImageProcessor


class LayoutService(BaseService):
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model = None
        self._model_lock = threading.Lock()
        self._initialized = True

    def _load(self):
        if self.model is not None:
            return
        with self._model_lock:
            if self.model is None:
                logger.info("[Layout] Loading YOLO model...")
                self.model = YOLOv10(settings.DOC_LAYOUT_MODEL_PATH)
                logger.info("[Layout] Model ready")

    async def process(self, file_content: bytes, filename: str) -> dict:
        await asyncio.to_thread(self._load)

        start = time.time()
        tmp_path = None

        try:
            tmp_path, original_size, scale = await asyncio.to_thread(
                ImageProcessor.preprocess, file_content
            )

            def _predict_locked():
                with self._model_lock:
                    return self.model.predict(
                        tmp_path, imgsz=1024, conf=0.25, verbose=False
                    )

            results = await asyncio.to_thread(_predict_locked)

            regions = self._extract_regions(results, scale)
            elapsed = round(time.time() - start, 2)

            return self._build_response(
                {
                    "filename": filename,
                    "image_count": len(regions),
                    "images": regions,
                    "processing_time": f"{elapsed}s",
                }
            )

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_regions(self, results, scale: float) -> list:
        regions = []
        if not results or not results[0].boxes:
            return regions

        result = results[0]
        figure_mask = result.boxes.cls == 3

        if not figure_mask.any():
            return regions

        boxes = result.boxes.xyxy[figure_mask].cpu().numpy()
        confs = result.boxes.conf[figure_mask].cpu().numpy()

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box)
            regions.append(
                {
                    "label": "image",
                    "bbox": [
                        int(x1 / scale),
                        int(y1 / scale),
                        int(x2 / scale),
                        int(y2 / scale),
                    ],
                    "confidence": round(float(conf), 4),
                }
            )

        return regions
