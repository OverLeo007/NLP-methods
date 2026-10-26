package ru.nlp.laba3.evaluation;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.evaluation.RelevanceJudgment;
import ru.nlp.laba3.model.evaluation.SubsetParameters;
import ru.nlp.laba3.model.evaluation.SubsetStatistics;
import ru.nlp.laba3.preprocessing.TextPreprocessor;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Отбор с приоритетом релевантных статей; память ограничена фоном и размеченными ID. */
@RequiredArgsConstructor
@Slf4j
final class RiaSubsetSelector {

    private final SubsetParameters parameters;

    private final TextPreprocessor preprocessor;

    private final Set<String> targetIds;

    private final Map<String, Document> relevant = new HashMap<>();

    private final Map<String, Document> background = new LinkedHashMap<>();

    private final Set<String> backgroundBodies = new HashSet<>();

    private long scanned;

    private long empty;

    private long duplicates;

    boolean accept(Document document) {
        scanned++;

        if (scanned % 100000 == 0) {
            log.info(
                    "[РИА] Просмотрено статей={}, найдено размеченных={}, фон={}",
                    scanned,
                    relevant.size(),
                    background.size()
            );
        }

        if (document.id() == null || document.id().isBlank()) {
            throw new InvalidInputException("Пустой ID статьи, запись " + scanned);
        }
        boolean needed = targetIds.contains(document.id()) && !relevant.containsKey(document.id());

        if (!needed && background.size() >= parameters.documents()) {
            return true;
        }

        if (preprocessor.tokens(document.text()).isEmpty()) {
            empty++;

            return true;
        }

        if (needed) {
            relevant.put(document.id(), document);
        }

        if (background.size() < parameters.documents()) {
            if (background.containsKey(document.id()) || !backgroundBodies.add(bodyKey(document.text()))) {
                duplicates++;
            } else {
                background.put(document.id(), document);
            }
        }

        return true;
    }

    Result select(List<EvaluationQuery> queries, Map<String, List<RelevanceJudgment>> judgments) {
        var shuffled = new ArrayList<>(queries);
        shuffled.sort(Comparator.comparing(EvaluationQuery::id));
        Collections.shuffle(shuffled, new Random(parameters.seed()));
        var selectedQueries = new ArrayList<EvaluationQuery>();
        var selectedJudgments = new ArrayList<RelevanceJudgment>();
        var documents = new LinkedHashMap<String, Document>();
        var bodies = new HashMap<String, String>();
        var queryTexts = new HashSet<String>();
        var excluded = new HashMap<String, Integer>();
        long titleMatches = 0;
        long bodyMatches = 0;

        for (var query : shuffled) {
            var relations = judgments.getOrDefault(query.id(), List.of());
            String reason = exclusion(query, relations, documents, bodies, queryTexts);

            if (reason == null && selectedQueries.size() >= parameters.queries()) {
                reason = "query_limit";
            }

            if (reason != null) {
                excluded.merge(reason, 1, Integer::sum);
                continue;
            }
            selectedQueries.add(query);
            selectedJudgments.addAll(relations);
            queryTexts.add(comparisonKey(query.text()));
            var related = relations.stream().map(row -> relevant.get(row.documentId())).toList();
            titleMatches += related.stream()
                    .anyMatch(doc -> comparisonKey(doc.title()).equals(comparisonKey(query.text()))) ? 1 : 0;
            bodyMatches += related.stream()
                    .anyMatch(doc -> comparisonKey(doc.text()).equals(comparisonKey(query.text()))) ? 1 : 0;

            for (var document : related) {
                documents.putIfAbsent(document.id(), document);
                bodies.put(bodyKey(document.text()), document.id());
            }
        }

        if (selectedQueries.size() != parameters.queries()) {
            throw new InvalidInputException(
                    "Недостаточно пригодных запросов: " + selectedQueries.size()
                            + " из " + parameters.queries() + "; причины исключения: " + excluded
            );
        }
        int relevantCount = documents.size();

        for (var document : background.values()) {
            if (documents.size() == parameters.documents()) {
                break;
            }

            if (!documents.containsKey(document.id()) && !bodies.containsKey(bodyKey(document.text()))) {
                documents.put(document.id(), document);
                bodies.put(bodyKey(document.text()), document.id());
            }
        }

        if (documents.size() != parameters.documents()) {
            throw new InvalidInputException(
                    "Недостаточно уникальных непустых статей: " + documents.size()
                            + " из " + parameters.documents()
            );
        }
        var statistics = SubsetStatistics.builder()
                .sourceQueries(queries.size())
                .queries(selectedQueries.size())
                .judgments(selectedJudgments.size())
                .documents(documents.size())
                .relevantDocuments(relevantCount)
                .scannedDocuments(scanned)
                .emptyCandidates(empty)
                .duplicateBackgroundCandidates(duplicates)
                .queryTitleMatches(titleMatches)
                .queryBodyMatches(bodyMatches)
                .excludedQueries(excluded)
                .build();

        return new Result(
                List.copyOf(documents.values()),
                List.copyOf(selectedQueries),
                List.copyOf(selectedJudgments),
                statistics
        );
    }

    private String exclusion(
            EvaluationQuery query,
            List<RelevanceJudgment> relations,
            Map<String, Document> documents,
            Map<String, String> bodies,
            Set<String> queryTexts
    ) {
        if (preprocessor.tokens(query.text()).isEmpty()) {
            return "empty_query";
        }

        if (relations.isEmpty()) {
            return "no_positive_judgment";
        }

        if (relations.stream().anyMatch(row -> !relevant.containsKey(row.documentId()))) {
            return "missing_or_empty_relevant_document";
        }

        if (queryTexts.contains(comparisonKey(query.text()))) {
            return "duplicate_query";
        }
        var additions = new HashMap<String, String>();
        int newDocuments = 0;

        for (var row : relations) {
            var document = relevant.get(row.documentId());
            String key = bodyKey(document.text());
            String owner = bodies.getOrDefault(key, additions.get(key));

            if (owner != null && !owner.equals(document.id())) {
                return "duplicate_relevant_body";
            }
            additions.put(key, document.id());

            if (!documents.containsKey(document.id())) {
                newDocuments++;
            }
        }

        return documents.size() + newDocuments > parameters.documents() ? "document_limit" : null;
    }

    private static String bodyKey(String text) {
        return Normalizer.normalize(text, Normalizer.Form.NFC).strip();
    }

    private static String comparisonKey(String text) {
        return Normalizer.normalize(text, Normalizer.Form.NFC).toLowerCase(Locale.ROOT).replace('ё', 'е').strip()
                .replaceAll("\\s+", " ");
    }

    /** Неизменяемый результат чистого отбора до записи файлов. */
    record Result(
            /** Статьи, вначале обязательные, затем фоновые. */
            List<Document> documents,
            /** Отложенные запросы. */
            List<EvaluationQuery> queries,
            /** Все положительные связи выбранных запросов. */
            List<RelevanceJudgment> judgments,
            /** Итоговые счётчики. */
            SubsetStatistics statistics
    ) {}
}
