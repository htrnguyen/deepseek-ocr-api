class TokenLoopError(Exception):
    """Raised when repetitive token output is detected during streaming."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class EmptyOutputError(Exception):
    """Raised when OCR output is suspiciously short (likely model hallucination)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
