import time
import os
import asyncio
import threading

from fastapi import HTTPException

from config import settings
from logger import logger
from paddleocr import TextDetection
from utils.image_processor import ImageProcessor

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


class PaddleDetectService:
    """Text detection service using PaddleOCR PP-OCRv5_server_det.

    Only runs the detection model (no recognition) → extremely fast.
    Returns polygon bounding boxes + confidence scores for every text region.
    """

    def __init__(self, model_name: str = "PP-OCRv5_server_det"):
        self._model = None
        self._model_name = model_name
        self._lock = threading.Lock()
        self._async_lock = None

    def _load_model(self):
        """Lazy-load the TextDetection model on first request (thread-safe)."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            logger.info(f"[_load_model] Lazy-loading TextDetection model: {self._model_name}")
            self._model = TextDetection(model_name=self._model_name)
        logger.info("[_load_model] TextDetection model loaded successfully")

    async def detect(
        self,
        file_content: bytes,
        filename: str,
    ) -> dict:
        """Detect text regions in an image. Returns boxes in original-image coordinates."""

        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        if self._model is None:
            await asyncio.to_thread(self._load_model)

        start_time = time.time()
        tmp_path = None

        try:
            try:
                tmp_path, original_size, scale = await asyncio.to_thread(
                    ImageProcessor.preprocess_image, file_content
                )
            except ValueError as e:
                logger.warning(
                    f"[detect] Image Rejected | File: {filename} | Error: {e}"
                )
                raise HTTPException(status_code=400, detail=str(e))

            logger.info(
                f"[detect] Image prepared | File: {filename} | "
                f"Original: {original_size[0]}x{original_size[1]} | Scale: {scale:.3f}"
            )

            det_start = time.time()
            async with self._async_lock:
                results = await asyncio.to_thread(
                    self._model.predict, tmp_path, batch_size=1
                )
            det_time = round(time.time() - det_start, 3)

            all_boxes = []
            for res in results:
                polys = res.get("dt_polys", [])
                scores = res.get("dt_scores", [])

                for i, poly in enumerate(polys):
                    score = scores[i] if i < len(scores) else 0.0

                    if scale < 1.0:
                        poly_orig = [[int(x / scale), int(y / scale)] for x, y in poly]
                    else:
                        poly_orig = [[int(x), int(y)] for x, y in poly]

                    xs = [p[0] for p in poly_orig]
                    ys = [p[1] for p in poly_orig]
                    bbox = [min(xs), min(ys), max(xs), max(ys)]

                    all_boxes.append(
                        {
                            "poly": poly_orig,
                            "bbox": bbox,
                            "score": round(float(score), 4),
                        }
                    )

            process_time = round(time.time() - start_time, 3)

            logger.info(
                f"[detect] Detection completed | File: {filename} | "
                f"Det time: {det_time}s | Total: {process_time}s | "
                f"Found: {len(all_boxes)} text region(s)"
            )

            return {
                "box_count": len(all_boxes),
                "boxes": all_boxes,
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "detection_time": f"{det_time}s",
                "processing_time": f"{process_time}s",
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
