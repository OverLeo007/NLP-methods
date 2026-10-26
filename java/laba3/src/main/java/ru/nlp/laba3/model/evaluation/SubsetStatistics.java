package ru.nlp.laba3.model.evaluation;

import java.util.Collections;
import java.util.Map;
import java.util.TreeMap;

import lombok.Builder;

/** Счётчики отбора; очистка относится только к проверенным кандидатам корпуса. */
@Builder
public record SubsetStatistics(
        /** Число исходных запросов. */
        int sourceQueries,
        /** Число выбранных запросов. */
        int queries,
        /** Число сохранённых положительных связей. */
        int judgments,
        /** Число выбранных статей. */
        int documents,
        /** Число обязательных релевантных статей в коллекции. */
        int relevantDocuments,
        /** Число прочитанных записей полного корпуса. */
        long scannedDocuments,
        /** Проверенные релевантные и фоновые кандидаты без токенов после нормализации. */
        long emptyCandidates,
        /** Отброшенные повторы ID или тела при наборе фонового префикса. */
        long duplicateBackgroundCandidates,
        /** Выбранные запросы, совпадающие с заголовком хотя бы одной релевантной статьи. */
        long queryTitleMatches,
        /** Выбранные запросы, совпадающие с телом хотя бы одной релевантной статьи. */
        long queryBodyMatches,
        /** Причины исключения запросов, включая остаток сверх запрошенного лимита. */
        Map<String, Integer> excludedQueries
) {

    /** Защищает счётчики от внешнего изменения. */
    public SubsetStatistics {
        excludedQueries = Collections.unmodifiableMap(new TreeMap<>(excludedQueries));
    }
}
