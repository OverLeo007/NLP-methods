package ru.nlp.laba3.preprocessing.rules;

import ru.nlp.laba3.preprocessing.TextNormalizationRule;

/** Сведение русского ё к е после приведения регистра. */
public final class RussianYoRule implements TextNormalizationRule {

    /** Идентификатор преобразования ё. */
    @Override
    public String id() {
        return NormalizationRuleType.RUSSIAN_YO;
    }

    /** Заменяет ё на е. */
    @Override
    public String normalize(String text) {
        return text.replace('ё', 'е');
    }
}
