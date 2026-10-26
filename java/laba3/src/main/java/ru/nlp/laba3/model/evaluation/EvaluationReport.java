package ru.nlp.laba3.model.evaluation;

import java.util.List;
import java.util.Map;

/** Полный протокол единой оценки ранее обученных моделей. */
public record EvaluationReport(
        /** Идентификатор правил оценки. */
        String protocol,
        /** Состав и правила отбора общей коллекции. */
        RiaSubsetManifest collection,
        /** SHA-256 проверенных файлов коллекции и манифеста. */
        Map<String, String> collectionSha256,
        /** Среда текущей оценки, отдельно от среды обучения. */
        Map<String, String> evaluationVersions,
        /** Общие границы и порядок измерений. */
        String measurement,
        /** Результаты алгоритмов на одних и тех же запросах. */
        List<ModelEvaluation<?>> models
) {

    /** Защищает снимок протокола. */
    public EvaluationReport {
        collectionSha256 = Map.copyOf(collectionSha256);
        evaluationVersions = Map.copyOf(evaluationVersions);
        models = List.copyOf(models);
    }
}
