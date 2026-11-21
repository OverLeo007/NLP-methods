package ru.nlp.semantic.corpus.dto;

/** Документ: заголовок остаётся метаданными и не участвует в обучении. */
public record Document(
        /** Уникальный идентификатор документа. */
        String id,
        /** Заголовок документа. */
        String title,
        /** Исходный текст документа. */
        String text
) {}
