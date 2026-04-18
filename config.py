from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MAX_LONG_SIDE: int = 1280
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    OLLAMA_TIMEOUT: int = 30
    RATE_LIMIT: str = "60/minute"
    DEFAULT_PROMPT: str = "<|grounding|>Convert the document to clean markdown."
    OLLAMA_MODEL: str = "deepseek-ocr"
    DOC_LAYOUT_MODEL_PATH: str = (
        "DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
