package ru.nlp.laba3.model.word2vec;

import ru.nlp.laba3.exception.InvalidInputException;

import lombok.Builder;

/**
 * Параметры обучения Word2Vec, без подбора по оценочным запросам.
 */
@Builder
public record Word2VecParameters(
        /** Размерность вектора слова. */
        int dimensions,
        /** Число эпох обучения. */
        int epochs,
        /** Минимальное число вхождений слова. */
        int minCount,
        /** Размер контекстного окна. */
        int window,
        /** Начальное значение генератора случайных чисел. */
        long seed,
        /** Число потоков Word2Vec и ND4J. */
        int workers
) {

    /** Проверяет допустимость размеров модели и счётчиков обучения. */
    public Word2VecParameters {
        if (dimensions < 2 || epochs < 1 || minCount < 1 || window < 1 || workers < 1) {
            throw new InvalidInputException("Word2Vec: размерность >= 2; эпохи, min-count, окно и потоки >= 1");
        }
    }
}
