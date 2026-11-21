package ru.nlp.semantic.model.search;

import com.fasterxml.jackson.annotation.JsonIgnore;

/** Документ, найденный общим векторным поиском. */
public record SearchHit(
        /** Уникальный идентификатор документа. */
        String id,
        /** Заголовок документа. */
        String title,
        /** Косинусное сходство с запросом. */
        double cosine,
        /** Начало исходного текста документа. */
        String excerpt,
        /** За последним символом фрагмента в документе есть продолжение. */
        @JsonIgnore boolean excerptTruncated
) {

    /** Создаёт результат, который содержит полный текст. */
    public SearchHit(String id, String title, double cosine, String excerpt) {
        this(id, title, cosine, excerpt, false);
    }
}
