import time
import os
import asyncio

from doclayout_yolo import YOLOv10

from config import settings
from logger import logger
from utils.image_processor import ImageProcessor


class DocLayoutService:
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info("[doclayout_service] Lazy-loading DocLayout-YOLO model...")
            self.model = YOLOv10(settings.DOC_LAYOUT_MODEL_PATH)
            logger.info("[doclayout_service] DocLayout-YOLO model loaded successfully")

    async def detect_figures(self, file_content: bytes, filename: str):
        if self.model is None:
            await asyncio.to_thread(self._load_model)

        start_time = time.time()
        tmp_path = None

        try:
            tmp_path, _ = await asyncio.to_thread(
                ImageProcessor.preprocess_image, file_content
            )

            logger.info(
                f"[detect_figures] DocLayout-YOLO figure detection started - File: {filename}"
            )

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
