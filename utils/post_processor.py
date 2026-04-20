import re


class OCRPostProcessor:
    """Post-process OCR output to clean artifacts."""

    GROUNDING_PATTERN = re.compile(
        r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL
    )

    @classmethod
    def clean(cls, text: str) -> str:
        """Post-process OCR output to remove grounding artifacts based on official DeepSeek-OCR v2 logic.

        Grounding mode outputs: <|ref|>word<|/ref|><|det|>[[x,y,x,y]]<|/det|>
        """
        if not text:
            return ""

        matches = cls.GROUNDING_PATTERN.findall(text)

        for match in matches:
            if "<|ref|>image<|/ref|>" in match[0]:
                text = text.replace(match[0], "", 1)
            else:
                text = re.sub(rf"(?m)^[^\n]*{re.escape(match[0])}[^\n]*\n?", "", text)

        text = text.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")

        return text.strip()
