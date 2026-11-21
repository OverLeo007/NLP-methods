package ru.nlp.semantic.model.evaluation;

import java.nio.file.Path;
import java.util.Objects;

import ru.nlp.semantic.model.analysis.Algorithm;

/** Один элемент упорядоченного списка ранее обученных моделей для общей оценки. */
public record EvaluationTarget(
        /** Алгоритм для выбора backend. */
        Algorithm algorithm,
        /** Каталог полного комплекта модели. */
        Path directory
) {

    /** Проверяет обязательные поля без открытия файлов или обучения. */
    public EvaluationTarget {
        Objects.requireNonNull(algorithm, "Алгоритм модели");
        Objects.requireNonNull(directory, "Каталог модели");
    }
}
