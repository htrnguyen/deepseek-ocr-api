from pydantic import BaseModel
from typing import Any


class BaseResponse(BaseModel):
    success: bool = True
    status: str = "success"


class OCRResponse(BaseResponse):
    filename: str
    text: str
    processing_time: str
    tokens: int = 0


class HealthResponse(BaseResponse):
    glm_ocr: str = "enabled"
    doclayout_yolo: str = "enabled"
    paddle_detect: str = "enabled"
    translate: str = "enabled"
    ollama_status: Any = {}
    version: str = "3.0"
