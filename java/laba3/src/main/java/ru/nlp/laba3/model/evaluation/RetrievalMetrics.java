package ru.nlp.laba3.model.evaluation;

import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

/** Макрооценка на всех запросах и диагностика покрытия без уменьшения знаменателя. */
public record RetrievalMetrics(
        /** Полный знаменатель, включая неуспешные запросы. */
        int queries,
        /** Среднее Recall@5. */
        double recallAt5,
        /** Среднее reciprocal rank с ограничением десятью результатами. */
        double mrrAt10,
        /** Число запросов с хотя бы одним релевантным результатом в первых пяти. */
        long hitQueriesAt5,
        /** Число запросов с релевантным результатом в первых десяти. */
        long hitQueriesAt10,
        /** Счётчики статусов поисковых запросов. */
        Map<String, Long> statuses,
        /** Число запросов с известными токенами, но неполным покрытием. */
        long partialCoverageQueries,
        /** Известные вхождения токенов во всех запросах. */
        long knownTokens,
        /** Все вхождения токенов во всех запросах. */
        long totalTokens,
        /** Отношение известных вхождений ко всем; ноль при отсутствии токенов. */
        double tokenCoverage,
        /** Число запросов, имеющих хотя бы одну нулевую релевантную статью. */
        long queriesWithZeroRelevantDocuments,
        /** Счётчики причин потери Recall@5; запрос с полным recall не включается. */
        Map<String, Long> recallLossReasons,
        /** Сумма времени search для всех запросов. */
        double searchSeconds,
        /** Среднее время search на один запрос. */
        double meanSearchSeconds,
        /** Наблюдаемый p95 по правилу ближайшего верхнего ранга. */
        double p95SearchSeconds
) {

    /** Сохраняет отсортированные неизменяемые счётчики. */
    public RetrievalMetrics {
        statuses = Collections.unmodifiableMap(new TreeMap<>(statuses));
        recallLossReasons = Collections.unmodifiableMap(new TreeMap<>(recallLossReasons));
    }
}
