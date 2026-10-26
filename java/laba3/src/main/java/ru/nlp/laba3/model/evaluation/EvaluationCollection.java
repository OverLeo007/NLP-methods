package ru.nlp.laba3.model.evaluation;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import ru.nlp.laba3.model.corpus.Document;

/** Проверенная общая коллекция с полным набором отложенных запросов. */
public record EvaluationCollection(
        /** Манифест выбора документов и запросов. */
        RiaSubsetManifest manifest,
        /** Исходные документы в порядке коллекции. */
        List<Document> documents,
        /** Все запросы, без фильтрации по покрытию конкретной модели. */
        List<EvaluationQuery> queries,
        /** Положительные связи по ID запроса. */
        Map<String, List<RelevanceJudgment>> judgments,
        /** Контрольные суммы файлов коллекции, включая манифест. */
        Map<String, String> sha256
) {

    /** Создаёт неизменяемый снимок данных оценки. */
    public EvaluationCollection {
        documents = List.copyOf(documents);
        queries = List.copyOf(queries);
        judgments = judgments.entrySet().stream()
                .collect(Collectors.toUnmodifiableMap(Map.Entry::getKey, entry -> List.copyOf(entry.getValue())));
        sha256 = Map.copyOf(sha256);
    }
}
