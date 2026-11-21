package ru.nlp.semantic.preprocessing.rules;

import ru.nlp.semantic.preprocessing.TextNormalizationRule;

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
