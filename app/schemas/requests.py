from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    """JSON body cho `POST /translate`: văn bản và ngôn ngữ nguồn/đích."""

    text: str = Field(description="Nội dung cần dịch.")
    target_language: str = Field(
        description="Ngôn ngữ đích (ví dụ: English, Vietnamese). Đưa vào prompt Ollama."
    )
    source_language: str = Field(
        default="auto",
        description="Ngôn ngữ nguồn; mặc định `auto` để model tự suy từ ngữ cảnh.",
    )
