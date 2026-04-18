import time
import os
import tempfile
from pathlib import Path
from fastapi import HTTPException
from doclayout_yolo import YOLOv10
from config import settings
from logger import logger


class DocLayoutService:
    def __init__(self):
        logger.info("Loading DocLayout-YOLO model...")
        self.model = YOLOv10(settings.DOC_LAYOUT_MODEL_PATH)
        logger.info("DocLayout-YOLO model loaded successfully")

    async def detect_figures(self, file_content: bytes, filename: str):
        start_time = time.time()
        tmp_path = None

        try:
            suffix = Path(filename).suffix.lower()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp_path = tmp.name
            tmp.write(file_content)
            tmp.close()

            logger.info(f"DocLayout-YOLO figure detection started - File: {filename}")

            results = self.model.predict(tmp_path, imgsz=1024, conf=0.25, verbose=False)
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
                f"DocLayout-YOLO completed in {process_time}s - Found {len(image_regions)} figure(s)"
            )

            return {
                "image_count": len(image_regions),
                "images": image_regions,
                "processing_time": f"{process_time}s",
            }

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
