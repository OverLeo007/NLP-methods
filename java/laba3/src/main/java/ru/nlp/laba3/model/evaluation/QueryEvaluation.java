package ru.nlp.laba3.model.evaluation;

import java.util.List;

import ru.nlp.laba3.model.search.SearchHit;

/** Выдача и вклад одного запроса в итоговые метрики. */
public record QueryEvaluation(
        /** ID оценочного запроса. */
        String queryId,
        /** Исходный текст запроса. */
        String text,
        /** Статус нормализации и векторизации. */
        String status,
        /** Число известных вхождений токенов. */
        int knownTokens,
        /** Общее число вхождений токенов. */
        int totalTokens,
        /** Все положительно размеченные ID статей. */
        List<String> relevantIds,
        /** Релевантные статьи с нулевым вектором в данной модели. */
        List<String> zeroRelevantIds,
        /** Первые десять результатов, включая cosine и фрагменты. */
        List<SearchHit> hits,
        /** Доля всех релевантных документов, найденная в первых пяти результатах. */
        double recallAt5,
        /** Обратный ранг первого релевантного результата до десятого включительно. */
        double reciprocalRankAt10,
        /** Почему Recall@5 не достиг единицы: статус запроса, нулевые статьи или ранжирование. */
        String recallLossReason,
        /** Время одного полного session.search, включая обработку и ранжирование. */
        double searchSeconds
) {

    /** Отделяет опубликованную выдачу и разметку от рабочих списков. */
    public QueryEvaluation {
        relevantIds = List.copyOf(relevantIds);
        zeroRelevantIds = List.copyOf(zeroRelevantIds);
        hits = List.copyOf(hits);
    }
}
