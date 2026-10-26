package ru.nlp.laba3.preprocessing;

/** Детерминированное правило нормализации исходного текста до токенизации. */
public interface TextNormalizationRule {

    /** Стабильный идентификатор правила для проверки совместимости модели. */
    String id();

    /** Применяет правило к тексту. */
    String normalize(String text);
}
