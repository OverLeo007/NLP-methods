package ru.nlp.laba3.model.evaluation;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Связь запроса и статьи из разметки РИА. */
public record RelevanceJudgment(
        /** Идентификатор оценочного запроса. */
        @JsonProperty("query-id") String queryId,
        /** Идентификатор релевантного документа. */
        @JsonProperty("corpus-id") String documentId,
        /** Степень релевантности; положительное значение означает релевантный документ. */
        Integer score
) {}
