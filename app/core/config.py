from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    API_TITLE: str = "MacOS AI Aggregator Service"
    API_VERSION: str = "3.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    RATE_LIMIT: str = "60/minute"

    OLLAMA_MODEL: str = "glm-ocr"
    OLLAMA_TRANSLATE_MODEL: str = "translategemma:latest"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_NUM_PREDICT: int = 2048
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_REPEAT_PENALTY: float = 3.5
    OLLAMA_REPEAT_LAST_N: int = 64
    OLLAMA_TOP_K: int = 40
    OLLAMA_TOP_P: float = 0.9
    OLLAMA_KEEPALIVE_INTERVAL: int = 300

    PROMPT_MARKDOWN: str = "Extract all text from this image and format it in Markdown. Preserve layout, headings, lists, tables. No conversational text."
    DEFAULT_PROMPT: str = PROMPT_MARKDOWN

    DOC_LAYOUT_MODEL_PATH: str = "models/doclayout_yolo_docstructbench_imgsz1024.pt"


settings = Settings()
