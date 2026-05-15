import os
import asyncio
import time
import threading
import httpx
from fastapi import HTTPException
from app.core.config import settings
from app.core.logging import logger
from app.services.base import BaseService


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
        self._initialized = True

    async def process(self, file_content: bytes, filename: str) -> dict:
        start = time.time()

        # Call the new layout OCR API
        url = "http://203.162.234.214:8181/api/v1/ocr"

        try:
            files = {"files": (filename, file_content, "image/jpeg")}
            data = {"refine": "true"}

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, data=data, files=files)
                response.raise_for_status()
                result = response.json()

            boxes = self._extract_boxes(result)
            elapsed = round(time.time() - start, 2)

            return self._build_response(
                {
                    "filename": filename,
                    "box_count": len(boxes),
                    "boxes": boxes,
                    "processing_time": f"{elapsed}s",
                }
            )

        except Exception as e:
            logger.error(f"[Detection] API call failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _extract_boxes(self, api_response: dict) -> list:
        boxes = []

        def traverse(node):
            # Extract nodes that have bounding box info, skip Page structural nodes
            if (
                "polygon" in node
                and "bbox" in node
                and node.get("block_type") != "Page"
            ):
                boxes.append(
                    {
                        "poly": node["polygon"],
                        "bbox": node["bbox"],
                        "score": 0.99,  # API doesn't provide score, default to high confidence
                    }
                )
            for child in node.get("children", []):
                traverse(child)

        results = api_response.get("results", [])
        if results:
            for child in results[0].get("children", []):
                traverse(child)

        return boxes
