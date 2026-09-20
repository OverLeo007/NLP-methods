"""Небольшой токенизатор на стандартной библиотеке для контракта лабораторной работы."""

from __future__ import annotations

import re
import unicodedata
from typing import Final

_WORD_CONNECTORS: Final = frozenset("-‑'’")
_NUMBER_PATTERN: Final = re.compile(r"\d+(?:[.,]\d+)?", flags=re.UNICODE)


def _is_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L")


def _word_end(text: str, start: int) -> int:
    index = start + 1
    while index < len(text):
        if _is_letter(text[index]):
            index += 1
            continue
        if (
            text[index] in _WORD_CONNECTORS
            and index + 1 < len(text)
            and _is_letter(text[index + 1])
        ):
            index += 1
            continue
        break
    return index


def tokenize(text: str) -> list[str]:
    """Вернуть NFC-нормализованные слова и числа с сохранением порядка."""
    normalized = unicodedata.normalize("NFC", text)
    tokens: list[str] = []
    index = 0

    while index < len(normalized):
        if _is_letter(normalized[index]):
            end = _word_end(normalized, index)
            tokens.append(normalized[index:end])
            index = end
            continue

        number_match = _NUMBER_PATTERN.match(normalized, index)
        if number_match is not None:
            tokens.append(number_match.group(0))
            index = number_match.end()
            continue

        index += 1

    return tokens
