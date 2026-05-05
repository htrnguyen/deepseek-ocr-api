from pydantic import BaseModel


class TranslateRequest(BaseModel):
    text: str
    target_language: str
    source_language: str = "auto"
