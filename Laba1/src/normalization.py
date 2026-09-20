"""Наблюдаемые ветви нормализации текста и версионируемые лексические ресурсы."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Protocol

import pymorphy3
import snowballstemmer

from .tokenizer import tokenize

NormalizationMethod = Literal["none", "stem", "lemma"]
RESOURCE_DIRECTORY: Final = Path(__file__).with_name("resources")
_CYRILLIC_WORD: Final = re.compile(r"[а-я]+", flags=re.IGNORECASE)
_HYPHEN: Final = re.compile(r"([-‑])")


class NormalizationError(ValueError):
    """Ошибка нарушения контракта конфигурации или лексического ресурса."""


class ParseResult(Protocol):
    @property
    def normal_form(self) -> str: ...

    @property
    def is_known(self) -> bool: ...


class MorphAnalyzer(Protocol):
    def parse(self, word: str) -> Sequence[ParseResult]: ...


class _Stemmer(Protocol):
    def stemWord(self, word: str) -> str: ...


@dataclass(frozen=True)
class NormalizationConfig:
    method: NormalizationMethod = "none"
    remove_stopwords: bool = False
    use_synonyms: bool = False

    def __post_init__(self) -> None:
        if self.method not in {"none", "stem", "lemma"}:
            raise NormalizationError(f"Неизвестный метод нормализации: {self.method}")


@dataclass(frozen=True)
class VocabularyResult:
    original_tokens: tuple[str, ...]
    normalized_tokens: tuple[str, ...]
    frequencies: dict[str, int]
    active_steps: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Преобразовать результат в стабильную публичную JSON-структуру."""
        return {
            "original_tokens": list(self.original_tokens),
            "normalized_tokens": list(self.normalized_tokens),
            "frequencies": self.frequencies,
            "active_steps": list(self.active_steps),
        }


class TextNormalizer:
    """Применить одну морфологическую ветвь и независимые лексические переключатели."""

    def __init__(
        self,
        config: NormalizationConfig,
        *,
        resources_dir: Path = RESOURCE_DIRECTORY,
        morph_analyzer: MorphAnalyzer | None = None,
        stemmer: _Stemmer | None = None,
    ) -> None:
        self.config = config
        self._morph = morph_analyzer
        self._stemmer = stemmer
        self._lemma_cache: dict[str, str] = {}
        self._stopwords: frozenset[str] = frozenset()
        self._synonyms: dict[str, str] = {}
        self._synonym_group_count = 0

        if config.method == "lemma" and self._morph is None:
            self._morph = pymorphy3.MorphAnalyzer()
        if config.method == "stem" and self._stemmer is None:
            self._stemmer = snowballstemmer.stemmer("russian")
        if config.remove_stopwords:
            self._stopwords = self._load_stopwords(resources_dir / "stopwords_ru.txt")
        if config.use_synonyms:
            self._synonyms, self._synonym_group_count = self._load_synonyms(
                resources_dir / "synonyms_ru.json"
            )

    @property
    def stopwords(self) -> frozenset[str]:
        return self._stopwords

    @property
    def synonyms(self) -> Mapping[str, str]:
        return MappingProxyType(self._synonyms)

    @property
    def synonym_group_count(self) -> int:
        return self._synonym_group_count

    @property
    def lemma_cache_size(self) -> int:
        return len(self._lemma_cache)

    @staticmethod
    def _common_form(token: str) -> str:
        return token.lower().replace("ё", "е")

    def _normalize_segment(self, segment: str) -> str:
        if self.config.method == "none" or not _CYRILLIC_WORD.fullmatch(segment):
            return segment
        if self.config.method == "stem":
            if self._stemmer is None:
                raise AssertionError("Стеммер не инициализирован")
            return self._stemmer.stemWord(segment).replace("ё", "е")

        cached = self._lemma_cache.get(segment)
        if cached is not None:
            return cached
        if self._morph is None:
            raise AssertionError("Морфологический анализатор не инициализирован")
        parses = self._morph.parse(segment)
        best = parses[0] if parses else None
        lemma = (
            best.normal_form.replace("ё", "е") if best is not None and best.is_known else segment
        )
        self._lemma_cache[segment] = lemma
        return lemma

    def normalize_token(self, token: str) -> str:
        """Нормализовать готовый токен без лексической фильтрации."""
        common = self._common_form(token)
        parts = _HYPHEN.split(common)
        return "".join(
            part if _HYPHEN.fullmatch(part) else self._normalize_segment(part) for part in parts
        )

    def _resource_token(self, raw: str, *, resource: Path) -> str:
        tokens = tokenize(raw)
        if len(tokens) != 1 or tokens[0] != raw:
            raise NormalizationError(
                f"Ресурс {resource} содержит значение, не являющееся одним токеном: {raw!r}"
            )
        return self.normalize_token(raw)

    def _load_stopwords(self, path: Path) -> frozenset[str]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise NormalizationError(f"Не удалось прочитать стоп-слова {path}: {error}") from error
        raw_words = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
        if len(set(raw_words)) < 50:
            raise NormalizationError("Список стоп-слов должен содержать минимум 50 разных слов")
        normalized = frozenset(self._resource_token(word, resource=path) for word in raw_words)
        negations = {self.normalize_token(word) for word in ("не", "ни")}
        if normalized & negations:
            raise NormalizationError("Стоп-список не должен удалять отрицания «не» и «ни»")
        return normalized

    def _load_synonyms(self, path: Path) -> tuple[dict[str, str], int]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise NormalizationError(f"Не удалось прочитать синонимы {path}: {error}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
            raise NormalizationError("Ресурс синонимов должен содержать список groups")
        groups = payload["groups"]
        if len(groups) < 10:
            raise NormalizationError("Ресурс должен содержать минимум 10 синонимических групп")

        mapping: dict[str, str] = {}
        for group in groups:
            if not isinstance(group, dict):
                raise NormalizationError("Каждая синонимическая группа должна быть объектом")
            canonical = group.get("canonical")
            variants = group.get("variants")
            if not isinstance(canonical, str) or not isinstance(variants, list):
                raise NormalizationError("Группа синонимов должна задавать canonical и variants")
            if len(variants) < 2 or not all(isinstance(item, str) for item in variants):
                raise NormalizationError(
                    "Каждая группа должна содержать минимум две строковые замены"
                )
            normalized_canonical = self._resource_token(canonical, resource=path)
            for raw in [canonical, *variants]:
                normalized_variant = self._resource_token(raw, resource=path)
                previous = mapping.get(normalized_variant)
                if previous is not None and previous != normalized_canonical:
                    raise NormalizationError(
                        f"Синоним {raw!r} после нормализации входит в разные группы"
                    )
                mapping[normalized_variant] = normalized_canonical
        return mapping, len(groups)

    def normalize(self, text: str) -> VocabularyResult:
        """Токенизировать текст и вернуть все наблюдаемые артефакты словаря."""
        original = tuple(tokenize(text))
        normalized: list[str] = []
        for token in original:
            value = self.normalize_token(token)
            if self.config.remove_stopwords and value in self._stopwords:
                continue
            if self.config.use_synonyms:
                value = self._synonyms.get(value, value)
            normalized.append(value)

        frequencies = dict(sorted(Counter(normalized).items()))
        active_steps = ["unicode_nfc", "lowercase", "yo_to_e", f"method:{self.config.method}"]
        if self.config.remove_stopwords:
            active_steps.append("stopwords")
        if self.config.use_synonyms:
            active_steps.append("synonyms")
        return VocabularyResult(
            original_tokens=original,
            normalized_tokens=tuple(normalized),
            frequencies=frequencies,
            active_steps=tuple(active_steps),
        )
