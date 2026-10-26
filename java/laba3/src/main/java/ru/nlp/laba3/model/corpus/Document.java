package ru.nlp.laba3.model.corpus;

/** Документ: заголовок остаётся метаданными и не участвует в обучении. */
public record Document(
        /** Уникальный идентификатор документа. */
        String id,
        /** Заголовок документа. */
        String title,
        /** Исходный текст документа. */
        String text
) {}
