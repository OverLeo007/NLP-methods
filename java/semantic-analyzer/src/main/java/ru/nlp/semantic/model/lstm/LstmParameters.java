package ru.nlp.semantic.model.lstm;

import ru.nlp.semantic.exception.InvalidInputException;

/** Размеры автокодировщика и параметры обучения на префиксах документов. */
public record LstmParameters(
        /** Размер вектора кодировщика и скрытого состояния декодера. */
        int dimensions,
        /** Размер обучаемого вектора токена. */
        int embeddingDimensions,
        /** Предельная длина последовательности. */
        int maxLength,
        /** Максимальное число обычных терминов. */
        int maxFeatures,
        /** Минимальная частота термина. */
        int minCount,
        /** Число документов в пакете. */
        int batchSize,
        /** Число проходов по корпусу. */
        int epochs,
        /** Скорость обучения Adam. */
        double learningRate,
        /** Начальное значение для инициализации и перемешивания. */
        long seed,
        /** Число нативных потоков. */
        int workers
) {

    /** Проверяет параметры до чтения корпуса. */
    public LstmParameters {
        if (
            dimensions < 2 || dimensions > 512 || embeddingDimensions < 2 || embeddingDimensions > 512
                    || maxLength < 1 || maxLength > 1024 || maxFeatures < 2 || maxFeatures > 50000
                    || minCount < 1 || batchSize < 1 || batchSize > 128 || epochs < 1 || epochs > 10000
                    || !Double.isFinite(learningRate) || learningRate <= 0 || workers < 1
        ) {
            throw new InvalidInputException(
                    "LSTM: dimensions и embedding-dimensions от 2 до 512, max-length от 1 до 1024, "
                            + "max-features от 2 до 50000, batch-size от 1 до 128, epochs от 1 до 10000, "
                            + "min-count и workers >= 1, learning-rate > 0 и конечна"
            );
        }
    }
}
