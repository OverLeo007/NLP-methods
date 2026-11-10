package ru.nlp.laba3.preprocessing.rules;

import static ru.nlp.laba3.preprocessing.rules.NormalizationRuleType.LOWERCASE;
import static ru.nlp.laba3.preprocessing.rules.NormalizationRuleType.NFC;
import static ru.nlp.laba3.preprocessing.rules.NormalizationRuleType.RUSSIAN_YO;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.preprocessing.TextNormalizationRule;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Фабрика неизменяемых правил по строковому идентификатору преобразования. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class NormalizationRuleFactory {

    /** Создаёт правило выбранного типа. */
    public static TextNormalizationRule create(String type) {
        if (type == null) {
            throw new InvalidInputException("Не задан идентификатор правила нормализации");
        }

        return switch (type) {
            case NFC -> new UnicodeNfcRule();
            case LOWERCASE -> new LowercaseRule();
            case RUSSIAN_YO -> new RussianYoRule();
            default -> throw new InvalidInputException("Неизвестное правило нормализации: " + type);
        };
    }
}
