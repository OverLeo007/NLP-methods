package ru.nlp.laba3.model.lsa;

import ru.nlp.laba3.exception.InvalidInputException;

/** Параметры обучения LSA; фактическая размерность ограничивается размером корпуса. */
public record LsaParameters(
        /** Запрошенное число SVD-компонент. */
        int dimensions,
        /** Максимальное число терминов по суммарной частоте в корпусе. */
        int maxFeatures,
        /** Число потоков нативного backend. */
        int workers
) {

    /** Проверяет границы параметров до чтения корпуса. */
    public LsaParameters {
        if (dimensions < 1 || maxFeatures < 2 || workers < 1) {
            throw new InvalidInputException("LSA: dimensions >= 1, max-features >= 2, workers >= 1");
        }
    }
}
