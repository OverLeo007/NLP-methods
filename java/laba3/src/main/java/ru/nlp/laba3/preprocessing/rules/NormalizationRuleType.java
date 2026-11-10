package ru.nlp.laba3.preprocessing.rules;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
/** Строковые идентификаторы правил, сохраняемые в профиле нормализации. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class NormalizationRuleType {

    /** Каноническая композиция Unicode. */
    public static final String NFC = "nfc";

    /** Нижний регистр Locale.ROOT. */
    public static final String LOWERCASE = "lower";

    /** Замена русского ё на е. */
    public static final String RUSSIAN_YO = "yo";
}
