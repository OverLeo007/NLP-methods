package ru.nlp.laba3.preprocessing.rules;

import java.util.Locale;

import ru.nlp.laba3.preprocessing.TextNormalizationRule;

/** Приведение регистра без зависимости от языка операционной системы. */
public final class LowercaseRule implements TextNormalizationRule {

    /** Идентификатор преобразования регистра. */
    @Override
    public String id() {
        return NormalizationRuleType.LOWERCASE;
    }

    /** Приводит текст к нижнему регистру. */
    @Override
    public String normalize(String text) {
        return text.toLowerCase(Locale.ROOT);
    }
}
