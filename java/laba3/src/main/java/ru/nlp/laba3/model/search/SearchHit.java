package ru.nlp.laba3.model.search;

/** Документ, найденный общим векторным поиском. */
public record SearchHit(
        /** Уникальный идентификатор документа. */
        String id,
        /** Заголовок документа. */
        String title,
        /** Косинусное сходство с запросом. */
        double cosine,
        /** Начало исходного текста документа. */
        String excerpt
) {}
