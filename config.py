from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file.

    DeepSeek OCR Reference (Ollama):
    - Model: deepseek-ocr:3b (6.7GB, 8K context)
    - Prompts: "Free OCR." | "<|grounding|>Convert the document to markdown."
    - Resolution modes: Tiny(512), Small(640), Base(1024), Large(1280)
    """

    MAX_LONG_SIDE: int = 1024
    MAX_FILE_SIZE: int = 10 * 1024 * 1024
    RATE_LIMIT: str = "60/minute"

    # --- Ollama model ---
    OLLAMA_MODEL: str = "deepseek-ocr"
    OLLAMA_TIMEOUT: int = 120
    OLLAMA_NUM_CTX: int = 8192
    OLLAMA_NUM_PREDICT: int = 4096
    OLLAMA_TEMPERATURE: float = 0.0
    OLLAMA_REPEAT_PENALTY: float = 1.15
    OLLAMA_REPEAT_LAST_N: int = -1  # Scan toàn bộ context để chống loop
    OLLAMA_TOP_K: int = 40
    OLLAMA_TOP_P: float = 0.9

    # --- Keepalive (giữ model loaded 24/7) ---
    OLLAMA_KEEPALIVE_INTERVAL: int = 300

    # --- OCR prompts ---
    # "Free OCR." — plain text extraction, tốt cho handwritten
    # "<|grounding|>Convert the document to markdown." — structured layout, tables
    PROMPT_FREE_OCR: str = "Free OCR."
    PROMPT_MARKDOWN: str = "<|grounding|>Convert the document to markdown."
    DEFAULT_PROMPT: str = "<|grounding|>Convert the document to markdown."

    # --- DocLayout ---
    DOC_LAYOUT_MODEL_PATH: str = (
        "DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1024.pt"
    )

    model_config = ConfigDict(env_file=".env", extra="ignore")


settings = Settings()
