import re
from logger import logger


class LoopDetector:
    """Detect repetitive token output during streaming."""

    LOOP_PATTERNS = [
        re.compile(r"(<td>\s*</td>){20,}"),
        re.compile(r"(<tr>\s*</tr>){15,}"),
        re.compile(r"(\|\s*){20,}"),
        re.compile(r"(</td><td>){20,}"),
        re.compile(r"(<td></td>){20,}"),
    ]

    @classmethod
    def detect(cls, tokens: list[str]) -> bool:
        """Detect repetitive output."""
        is_pure_structure = False
        recent_tail = "".join(tokens[-50:])
        if len(recent_tail) >= 30:
            clean_tail = re.sub(r'<[^>]+>|\s|\||-', '', recent_tail)
            if len(clean_tail) == 0:
                is_pure_structure = True

        recent_text = "".join(tokens[-150:]) if len(tokens) >= 150 else "".join(tokens)

        if not is_pure_structure:
            window = tokens[-100:] if len(tokens) >= 100 else tokens
            if len(window) >= 40:
                unique_ratio = len(set(window)) / len(window)
                if unique_ratio < 0.25:
                    logger.warning(
                        f"[loop_detect] Strategy 1 (Unique Ratio) | {unique_ratio:.2%} < 25% | "
                        f"Tokens: {len(set(window))}/{len(window)} | "
                        f"Sample: {''.join(window[-10:])}"
                    )
                    return True

            if len(tokens) >= 60:
                recent = "".join(tokens[-30:])
                earlier = "".join(tokens[-60:-30])
                if recent == earlier and len(recent) > 10:
                    logger.warning(
                        f"[loop_detect] Strategy 2 (Window Match) | "
                        f"30-token window repeated | Match: '{recent[:30]}...'"
                    )
                    return True

            if len(recent_text) >= 50:
                unique_chars = len(set(recent_text))
                if unique_chars < 15:
                    logger.warning(
                        f"[loop_detect] Strategy 3a (Char Uniqueness) | "
                        f"Chars: {unique_chars}/{len(recent_text)} | "
                        f"Sample: '{recent_text[-30:]}'"
                    )
                    return True

            if len(recent_text) >= 60:
                for p_len in range(15, 40):
                    pattern = recent_text[-p_len:]
                    if recent_text.endswith(pattern * 3):
                        clean_pattern = re.sub(r'<[^>]+>|\s|\||-', '', pattern)
                        if len(clean_pattern) == 0:
                            if recent_text.endswith(pattern * 25):
                                logger.warning(
                                    f"[loop_detect] Strategy 3b (Table Repeat) | "
                                    f"Pattern: '{pattern}' repeated 25+ times"
                                )
                                return True
                        else:
                            logger.warning(
                                f"[loop_detect] Strategy 3b (Substring Repeat) | "
                                f"Pattern: '{pattern}' (len {p_len}) repeated 3+ times"
                            )
                            return True

        for pattern in cls.LOOP_PATTERNS:
            if pattern.search(recent_text):
                logger.warning(
                    f"[loop_detect] Strategy 4 (Regex Pattern) | "
                    f"Match: {pattern.pattern} | Text: '{recent_text[-40:]}'"
                )
                return True

        return False
