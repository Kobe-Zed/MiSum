from __future__ import annotations

import re
from typing import List


TOKEN_PATTERN = re.compile(
    r"<[^>]+>|[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|->|::|&&|\|\||[^\s]"
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ").strip())


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    normalized = normalize_text(text)
    if lowercase:
        normalized = normalized.lower()
    return TOKEN_PATTERN.findall(normalized)
