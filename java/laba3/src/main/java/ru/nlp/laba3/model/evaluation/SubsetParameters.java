package ru.nlp.laba3.model.evaluation;

import ru.nlp.laba3.exception.InvalidInputException;

/** Параметры воспроизводимого отбора оценочной коллекции. */
public record SubsetParameters(
        /** Требуемое число уникальных непустых статей. */
        int documents,
        /** Требуемое число пригодных запросов. */
        int queries,
        /** Seed перемешивания запросов после сортировки по ID. */
        long seed
) {

    /** Проверяет размеры до чтения исходных файлов. */
    public SubsetParameters {
        if (documents < 2 || queries < 1) {
            throw new InvalidInputException("Требуется не меньше 2 документов и 1 запроса");
        }
    }
}
