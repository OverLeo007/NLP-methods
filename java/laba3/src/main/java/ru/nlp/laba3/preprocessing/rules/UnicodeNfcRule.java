package ru.nlp.laba3.preprocessing.rules;

import java.text.Normalizer;

import ru.nlp.laba3.preprocessing.TextNormalizationRule;

/** Приведение составных Unicode-символов к NFC. */
public final class UnicodeNfcRule implements TextNormalizationRule {

    /** Идентификатор Unicode NFC. */
    @Override
    public String id() {
        return NormalizationRuleType.NFC;
    }

    /** Приводит текст к NFC. */
    @Override
    public String normalize(String text) {
        return Normalizer.normalize(text, Normalizer.Form.NFC);
    }
}
