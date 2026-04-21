import re


class OCRPostProcessor:
    """Post-process OCR output to clean artifacts."""

    GROUNDING_PATTERN = re.compile(
        r"(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)", re.DOTALL
    )
    BBOX_COORD_PATTERN = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")

    @classmethod
    def extract_bboxes(cls, text: str) -> list[dict]:
        """Extract bounding boxes from complete grounding tag pairs.

        Returns list of {"text": str, "bbox": [x1, y1, x2, y2]} dicts.
        Coordinates are in DeepSeek's normalized 0-999 space.
        """
        if not text:
            return []

        bboxes = []
        for _full, ref_content, det_content in cls.GROUNDING_PATTERN.findall(text):
            if ref_content.strip() == "image":
                continue

            for coord_match in cls.BBOX_COORD_PATTERN.finditer(det_content):
                x1, y1, x2, y2 = map(int, coord_match.groups())
                bboxes.append({
                    "text": ref_content.strip(),
                    "bbox": [x1, y1, x2, y2],
                })

        return bboxes

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
