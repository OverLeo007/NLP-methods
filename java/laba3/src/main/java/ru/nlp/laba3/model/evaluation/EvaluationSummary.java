package ru.nlp.laba3.model.evaluation;

import java.util.List;

import ru.nlp.laba3.model.analysis.Algorithm;

/** Краткий результат CLI; подробная выдача хранится в report.json. */
public record EvaluationSummary(
        /** Каталог нового оценочного протокола. */
        String output,
        /** Число запросов в знаменателе каждого алгоритма. */
        int queries,
        /** Сводки двух анализаторов. */
        List<ModelSummary> models
) {

    /** Фиксирует список сводок. */
    public EvaluationSummary {
        models = List.copyOf(models);
    }

    /** Метрики и нулевые документы одного анализатора. */
    public record ModelSummary(
            /** Алгоритм. */
            Algorithm algorithm,
            /** Нулевые документы. */
            long zeroDocuments,
            /** Полная сводка метрик. */
            RetrievalMetrics metrics
    ) {}
}
