package ru.nlp.laba3.preprocessing;

import java.util.List;
import java.util.regex.MatchResult;
import java.util.regex.Pattern;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Внутренний токенизатор уже нормализованного текста. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
final class TextTokenizer {

    private static final Pattern TOKEN = Pattern.compile("\\p{L}+(?:[-‑'’]\\p{L}+)*|\\p{Nd}+(?:[.,]\\p{Nd}+)?");

    static List<String> tokens(String normalizedText) {
        return TOKEN.matcher(normalizedText).results().map(MatchResult::group).toList();
    }
}
