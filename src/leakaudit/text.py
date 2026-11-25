"""Text normalisation, entity masking and sentence splitting.

Lower-casing is Turkish-aware (I -> ı, İ -> i) so that dotted and dotless
i do not split otherwise identical documents. Masking replaces URLs, e-mail
addresses, phone numbers and numbers with typed placeholders; this is the
step that reveals template collapse (thousands of distinct strings that are
one template with different links or amounts).
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MAIL_RE = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
TEL_RE = re.compile(r"(?:\+90|0)?[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}")
NUM_RE = re.compile(r"\b\d[\d.,]*\b")
WS_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")
PUNCT_RE = re.compile(r"[^\w\s]|_", re.UNICODE)
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")

_TR_UPPER = str.maketrans({"I": "ı", "İ": "i"})


def tr_lower(text: str) -> str:
    """Turkish-aware lower-casing."""
    return str(text).translate(_TR_UPPER).lower()


def mask(text: str) -> str:
    """Lower-case, mask entities and collapse whitespace."""
    t = tr_lower(text)
    t = URL_RE.sub(" [url] ", t)
    t = MAIL_RE.sub(" [email] ", t)
    t = TEL_RE.sub(" [phone] ", t)
    t = NUM_RE.sub(" [num] ", t)
    return WS_RE.sub(" ", t).strip()


def sentences(text: str, min_tokens: int = 4) -> List[str]:
    """Masked sentences with at least ``min_tokens`` tokens."""
    return [s.strip() for s in SENT_SPLIT_RE.split(mask(text))
            if len(s.strip().split(" ")) >= min_tokens]


def shingles(text: str, k: int = 5) -> Set[str]:
    """Character k-shingles of the masked text."""
    s = mask(text)
    if len(s) < k:
        return {s} if s else set()
    return {s[i:i + k] for i in range(len(s) - k + 1)}


def first_tokens(text: str, k: int) -> str:
    return " ".join(mask(text).split(" ")[:k])


def n_tokens(texts: Iterable[str]) -> List[int]:
    return [len(mask(t).split(" ")) for t in texts]
