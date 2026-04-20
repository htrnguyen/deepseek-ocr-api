import re
from logger import logger


class LoopDetector:
    """Detect repetitive token output during streaming."""

    LOOP_PATTERNS = [
        re.compile(r"(<td>\s*</td>){6,}"),
        re.compile(r"(<tr>\s*</tr>){4,}"),
        re.compile(r"(\|\s*){12,}"),
        re.compile(r"(</td><td>){6,}"),
        re.compile(r"(<td></td>){5,}"),
    ]

    @classmethod
    def detect(cls, tokens: list[str]) -> bool:
        """Detect repetitive output."""
        if len(tokens) < 20:
            return False

        window = tokens[-100:] if len(tokens) >= 100 else tokens
        if len(window) >= 40:
            unique_ratio = len(set(window)) / len(window)
            if unique_ratio < 0.25:
                logger.warning(
                    f"[loop_detect] Strategy 1 (Unique Ratio): {unique_ratio:.2%} < 25%. "
                    f"Tokens: {len(set(window))} unique / {len(window)} total. "
                    f"Sample: {''.join(window[-10:])}"
                )
                return True

        if len(tokens) >= 60:
            recent = "".join(tokens[-30:])
            earlier = "".join(tokens[-60:-30])
            if recent == earlier and len(recent) > 10:
                logger.warning(
                    f"[loop_detect] Strategy 2 (Window Match): "
                    f"30-token window repeated exactly. Match: '{recent[:30]}...'"
                )
                return True

        recent_text = "".join(tokens[-150:]) if len(tokens) >= 150 else "".join(tokens)

        if len(recent_text) >= 50:
            unique_chars = len(set(recent_text))
            if unique_chars < 15:
                logger.warning(
                    f"[loop_detect] Strategy 3a (Char Uniqueness): "
                    f"{unique_chars} unique chars in {len(recent_text)} length text. "
                    f"Sample: '{recent_text[-30:]}'"
                )
                return True

        if len(recent_text) >= 60:
            for p_len in range(15, 40):
                pattern = recent_text[-p_len:]
                if recent_text.endswith(pattern * 3):
                    logger.warning(
                        f"[loop_detect] Strategy 3b (Substring Repeat): "
                        f"Pattern '{pattern}' (len {p_len}) repeated 3+ times at end."
                    )
                    return True

        for pattern in cls.LOOP_PATTERNS:
            if pattern.search(recent_text):
                logger.warning(
                    f"[loop_detect] Strategy 4 (Regex Pattern): "
                    f"Matched pattern {pattern.pattern} on text: '{recent_text[-40:]}'"
                )
                return True

        return False
