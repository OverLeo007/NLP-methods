from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import pytest

from Laba1.src import cli
from Laba1.src.normalization import (
    RESOURCE_DIRECTORY,
    MorphAnalyzer,
    NormalizationConfig,
    NormalizationMethod,
    ParseResult,
    TextNormalizer,
)
from Laba1.src.tokenizer import tokenize


def test_tokenize_handles_unicode_words_numbers_and_internal_separators() -> None:
    text = "Е\u0308ЛКА, Мир_world 12 3,14 2.5 по‑русски rock’n’roll x--y 🙂"

    assert tokenize(text) == [
        "ЁЛКА",
        "Мир",
        "world",
        "12",
        "3,14",
        "2.5",
        "по‑русски",
        "rock’n’roll",
        "x",
        "y",
    ]


@pytest.mark.parametrize("text", ["", "___ ... 🙂"])
def test_tokenize_returns_empty_list_for_empty_or_delimiter_only_input(text: str) -> None:
    assert tokenize(text) == []


def test_tokenize_accepts_unicode_letters_but_rejects_other_numeric_categories() -> None:
    text = "² ½ Ⅻ ௰ Ελληνικά 漢字 ქართული १२ 1,5"

    assert tokenize(text) == ["Ελληνικά", "漢字", "ქართული", "१२", "1,5"]


@pytest.fixture
def normalizers() -> dict[str, TextNormalizer]:
    return {
        method: TextNormalizer(NormalizationConfig(method=method))
        for method in ("none", "stem", "lemma")
    }


def test_common_normalization_preserves_latin_numbers_and_unknown_words(
    normalizers: dict[str, TextNormalizer],
) -> None:
    source = "ЁЖИК HELLO 42 крокозябра"

    assert normalizers["none"].normalize(source).normalized_tokens == (
        "ежик",
        "hello",
        "42",
        "крокозябра",
    )
    assert normalizers["lemma"].normalize(source).normalized_tokens == (
        "ежик",
        "hello",
        "42",
        "крокозябра",
    )


def test_stem_and_lemma_are_alternatives_and_preserve_hyphens(
    normalizers: dict[str, TextNormalizer],
) -> None:
    source = "Платья русско-английские русско‑английские"

    assert normalizers["stem"].normalize(source).normalized_tokens == (
        "плат",
        "русск-английск",
        "русск‑английск",
    )
    assert normalizers["lemma"].normalize(source).normalized_tokens == (
        "платье",
        "русско-английский",
        "русско‑английский",
    )


@pytest.mark.parametrize(
    ("stopwords", "synonyms", "expected"),
    [
        (False, False, ("это", "отличный", "платье", "и", "прекрасный", "куртка")),
        (True, False, ("это", "отличный", "платье", "прекрасный", "куртка")),
        (False, True, ("это", "хороший", "платье", "и", "хороший", "куртка")),
        (True, True, ("это", "хороший", "платье", "хороший", "куртка")),
    ],
)
def test_stopwords_and_synonyms_are_independent_switches(
    stopwords: bool,
    synonyms: bool,
    expected: tuple[str, ...],
) -> None:
    normalizer = TextNormalizer(
        NormalizationConfig(
            method="lemma",
            remove_stopwords=stopwords,
            use_synonyms=synonyms,
        )
    )

    result = normalizer.normalize("Это отличное платье и прекрасная куртка")
    assert result.normalized_tokens == expected


@pytest.mark.parametrize(("method", "expected"), [("stem", "хорош"), ("lemma", "хороший")])
def test_resources_meet_minimums_preserve_negations_and_use_selected_branch(
    method: NormalizationMethod, expected: str
) -> None:
    normalizer = TextNormalizer(
        NormalizationConfig(method=method, remove_stopwords=True, use_synonyms=True)
    )

    assert len(normalizer.stopwords) >= 50
    assert normalizer.normalize_token("не") not in normalizer.stopwords
    assert normalizer.normalize_token("ни") not in normalizer.stopwords
    assert normalizer.synonym_group_count == 10
    assert normalizer.normalize("ЕЩЁ прекрасная").normalized_tokens == (expected,)


def test_synonym_resource_records_dictionary_source_for_every_group() -> None:
    payload = json.loads((RESOURCE_DIRECTORY / "synonyms_ru.json").read_text(encoding="utf-8"))

    assert payload["metadata"]["origin"].startswith("Ограниченная учебная выборка")
    assert "RuReviews" in payload["metadata"]["rule"]
    assert len(payload["groups"]) == 10
    assert all(group["source"].startswith("https://kartaslov.ru/") for group in payload["groups"])


def test_repeated_lemmas_are_cached() -> None:
    class CountingMorph(MorphAnalyzer):
        def __init__(self) -> None:
            self.calls = 0
            self.words: list[str] = []

        def parse(self, word: str) -> Sequence[ParseResult]:
            self.calls += 1
            self.words.append(word)
            return [cast(ParseResult, FakeParse(normal_form="платье", is_known=True))]

    morph = CountingMorph()
    normalizer = TextNormalizer(NormalizationConfig(method="lemma"), morph_analyzer=morph)

    assert normalizer.normalize("Платья платья").normalized_tokens == ("платье", "платье")
    assert morph.calls == 1
    assert normalizer.lemma_cache_size == 1


def test_result_exposes_tokens_frequencies_and_active_steps(
    normalizers: dict[str, TextNormalizer],
) -> None:
    result = normalizers["none"].normalize("Платье, платье!")

    assert result.original_tokens == ("Платье", "платье")
    assert result.frequencies == {"платье": 2}
    assert result.active_steps == ("unicode_nfc", "lowercase", "yo_to_e", "method:none")

    empty = normalizers["none"].normalize("___ ... 🙂")
    assert empty.original_tokens == ()
    assert empty.normalized_tokens == ()
    assert empty.frequencies == {}


def test_vocabulary_command_prints_stable_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(
        [
            "vocabulary",
            "--text",
            "Очень хорошее платье",
            "--method",
            "lemma",
            "--stopwords",
            "--synonyms",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["original_tokens"] == ["Очень", "хорошее", "платье"]
    assert payload["normalized_tokens"] == ["очень", "хороший", "платье"]
    assert payload["frequencies"] == {"очень": 1, "платье": 1, "хороший": 1}
    assert payload["active_steps"] == [
        "unicode_nfc",
        "lowercase",
        "yo_to_e",
        "method:lemma",
        "stopwords",
        "synonyms",
    ]


@dataclass(frozen=True)
class FakeParse:
    normal_form: str
    is_known: bool
