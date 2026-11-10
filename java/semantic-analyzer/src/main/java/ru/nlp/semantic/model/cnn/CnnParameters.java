package ru.nlp.semantic.model.cnn;

import ru.nlp.semantic.cnn.PairSampling;
import ru.nlp.semantic.exception.InvalidInputException;

import lombok.Builder;

/** Параметры CNN и контрастивного обучения без оценочных запросов. */
@Builder
public record CnnParameters(
        /** Размер выходного вектора текста. */
        int dimensions,
        /** Размер обучаемого вектора слова. */
        int embeddingDimensions,
        /** Число свёрточных фильтров. */
        int filters,
        /** Нечётная ширина свёртки со stride 1 и Same padding. */
        int kernelSize,
        /** Максимальная длина фрагмента и префикса при поиске. */
        int maxLength,
        /** Максимальное число обычных терминов, без PAD и OOV. */
        int maxFeatures,
        /** Минимальная частота термина. */
        int minCount,
        /** Число документов в пакете; каждый даёт два фрагмента. */
        int batchSize,
        /** Число проходов по обучающим документам. */
        int epochs,
        /** Скорость Adam. */
        double learningRate,
        /** Температура контрастивной функции потерь. */
        double temperature,
        /** Seed инициализации, перемешивания и выбора фрагментов. */
        long seed,
        /** Число потоков нативных вычислений. */
        int workers,
        /** Доля отдельных документов для validation; 0 означает обучение на всём корпусе. */
        double validationFraction,
        /** Выбор положительных фрагментов; отсутствующее поле старой модели означает symmetric. */
        PairSampling pairSampling
) {

    /** Проверяет размеры до чтения корпуса и выделения памяти. */
    public CnnParameters {
        pairSampling = pairSampling == null ? PairSampling.symmetric : pairSampling;

        if (
            dimensions < 2 || dimensions > 4096 || embeddingDimensions < 2 || embeddingDimensions > 1024
                    || filters < 2 || filters > 4096 || kernelSize < 1 || kernelSize > 31
                    || kernelSize % 2 == 0 || maxLength < kernelSize || maxLength > 4096
                    || maxFeatures < 2 || maxFeatures > 50_000 || minCount < 1
                    || batchSize < 2 || batchSize > 128 || epochs < 1 || epochs > 10_000 || workers < 1
                    || !Double.isFinite(learningRate) || learningRate <= 0
                    || !Double.isFinite(temperature) || temperature < 0.01 || temperature > 10
                    || !Double.isFinite(validationFraction) || validationFraction < 0 || validationFraction > 0.5
        ) {
            throw new InvalidInputException(
                    "CNN: dimensions, embedding-dimensions, filters, max-features >= 2; "
                            + "kernel-size нечётный >= 1 и <= max-length; min-count, epochs, workers >= 1; "
                            + "batch-size от 2 до 128; learning-rate > 0; temperature от 0.01 до 10; "
                            + "validation-fraction от 0 до 0.5. "
                            + "Верхние границы: dimensions/filters/max-length 4096, embedding-dimensions 1024, "
                            + "kernel-size 31, max-features 50000, epochs 10000"
            );
        }
        long weights = ((long) maxFeatures + 2) * embeddingDimensions
                + (long) filters * (embeddingDimensions * kernelSize + 1L)
                + (long) dimensions * (filters + 1L);
        long activations = 2L * (batchSize + 1) * maxLength * Math.max(embeddingDimensions, filters);

        if (weights > 5_000_000 || activations > 20_000_000) {
            throw new InvalidInputException("CNN: превышен лимит 5 000 000 весов или 20 000 000 активаций пакета");
        }
    }
}
