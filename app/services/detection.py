import os
import asyncio
import time
import threading
from fastapi import HTTPException
from paddleocr import TextDetection
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BaseService
from app.utils.image import ImageProcessor

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


class DetectionService(BaseService):
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
        self._model = None
        self._model_lock = threading.Lock()
        self._initialized = True

    def _load(self):
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is None:
                logger.info("[Detection] Loading Paddle model...")
                self._model = TextDetection(model_name="PP-OCRv5_server_det")
                logger.info("[Detection] Model ready")

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
                    return self._model.predict(tmp_path, batch_size=1)

            results = await asyncio.to_thread(_predict_locked)
            boxes = self._extract_boxes(results, scale)

            elapsed = round(time.time() - start, 2)

            return self._build_response({
                "filename": filename,
                "box_count": len(boxes),
                "boxes": boxes,
                "processing_time": f"{elapsed}s",
            })

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _extract_boxes(self, results, scale: float) -> list:
        boxes = []
        for res in results:
            polys = res.get("dt_polys", [])
            scores = res.get("dt_scores", [])

            for i, poly in enumerate(polys):
                score = scores[i] if i < len(scores) else 0.0
                poly_orig = [[int(x / scale), int(y / scale)] for x, y in poly] if scale < 1 else [[int(x), int(y)] for x, y in poly]
                xs, ys = [p[0] for p in poly_orig], [p[1] for p in poly_orig]

                boxes.append({
                    "poly": poly_orig,
                    "bbox": [min(xs), min(ys), max(xs), max(ys)],
                    "score": round(float(score), 4),
                })

        return boxes
