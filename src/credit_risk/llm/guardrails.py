from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

INJECTION_PATTERNS = {
    "override": r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|rules?|prompt|above|previous)\b",
    "role_change": r"\b(you are now|act as|pretend to be|from now on you)\b",
    "prompt_leak": r"\b(system prompt|reveal|print|show)\b.{0,30}\b(prompt|instructions?|rules?)\b",
    "decision_steer": r"\b(approve|mark)\b.{0,30}\b(this|my|the)\b.{0,20}\b(application|loan|applicant)\b",
    "delimiter": r"(```|<\s*/?\s*(system|assistant|user)\s*>|\[/?INST\])",
}
_COMPILED = {k: re.compile(v, re.I | re.S) for k, v in INJECTION_PATTERNS.items()}


@dataclass
class ScreenResult:
    text: str
    flags: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return bool(self.flags)


def sanitize(text: str, max_chars: int = 500) -> str:
    """Normalize unicode, drop control characters and cap length."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C"
    )
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def screen_user_text(text: str, max_chars: int = 500) -> ScreenResult:
    """Sanitize applicant-supplied free text and flag injection attempts.

    Flagged text is not sent to the model; the case goes to manual review.
    """
    clean = sanitize(text, max_chars)
    flags = [name for name, pattern in _COMPILED.items() if pattern.search(clean)]
    return ScreenResult(clean, flags)
