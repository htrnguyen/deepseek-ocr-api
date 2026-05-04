import time
import os
import asyncio
import threading

from doclayout_yolo import YOLOv10
from fastapi import HTTPException

from config import settings
from logger import logger
from utils.image_processor import ImageProcessor


class DocLayoutService:
    def __init__(self):
        self.model = None
        self._lock = threading.Lock()
        self._async_lock = None

    def _load_model(self):
        if self.model is not None:
            return

        with self._lock:
            if self.model is not None:
                return

            logger.info("[_load_model] Lazy-loading DocLayout-YOLO model")
            self.model = YOLOv10(settings.DOC_LAYOUT_MODEL_PATH)
            logger.info("[_load_model] DocLayout-YOLO model loaded successfully")

    async def detect_figures(self, file_content: bytes, filename: str):
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        if self.model is None:
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
                    f"[detect_figures] | Image Rejected | File: {filename} | {e}"
                )
                raise HTTPException(status_code=400, detail=str(e))

            logger.info(f"[detect_figures] | DocLayout-YOLO started | File: {filename}")

            async with self._async_lock:
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
                            x1_r, y1_r, x2_r, y2_r = map(int, box)

                            x1 = int(x1_r / scale)
                            y1 = int(y1_r / scale)
                            x2 = int(x2_r / scale)
                            y2 = int(y2_r / scale)

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
                f"[detect_figures] | DocLayout-YOLO completed | "
                f"Time: {process_time}s | Found: {len(image_regions)} figure(s)"
            )

            return {
                "image_count": len(image_regions),
                "images": image_regions,
                "processing_time": f"{process_time}s",
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
