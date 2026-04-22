from pydantic_settings import BaseSettings
from pydantic import ConfigDict


from typing import List


class Settings(BaseSettings):
    """Application settings"""

    API_TITLE: str = "GLM OCR API"
    API_VERSION: str = "2.5"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    RATE_LIMIT: str = "60/minute"

    OLLAMA_MODEL: str = "glm-ocr"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_NUM_PREDICT: int = 2048
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_REPEAT_PENALTY: float = 3.5
    OLLAMA_REPEAT_LAST_N: int = 64
    OLLAMA_TOP_K: int = 40
    OLLAMA_TOP_P: float = 0.9

    OLLAMA_KEEPALIVE_INTERVAL: int = 300
    PROMPT_FREE_OCR: str = "Extract the text from this image."
    PROMPT_MARKDOWN: str = (
        "Please extract all text from this image and format it meticulously in Markdown. Preserve the original layout, headings, lists, and tables exactly as they appear. Do not add any conversational text."
    )
    PROMPT_GENERAL_OCR: str = "Please perform OCR on this image."
    DEFAULT_PROMPT: str = PROMPT_MARKDOWN

    DOC_LAYOUT_MODEL_PATH: str = (
        "DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"
    )

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()
