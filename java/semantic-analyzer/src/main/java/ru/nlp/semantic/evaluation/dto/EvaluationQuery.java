package ru.nlp.semantic.evaluation.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

/** Отложенный запрос, который не передаётся обучению. */
public record EvaluationQuery(
        /** Идентификатор запроса в исходном наборе РИА. */
        @JsonProperty("_id") String id,
        /** Исходный текст запроса. */
        String text
) {}
