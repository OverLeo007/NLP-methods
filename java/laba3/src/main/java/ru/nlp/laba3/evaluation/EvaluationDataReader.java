package ru.nlp.laba3.evaluation;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.evaluation.RelevanceJudgment;

import lombok.RequiredArgsConstructor;

/** Единое чтение запросов и разметки для подготовки коллекции и её оценки. */
@Component
@RequiredArgsConstructor
public class EvaluationDataReader {

    private final ObjectMapper jsonMapper;

    /** Читает запросы, проверяя поля и уникальность ID без фильтрации по токенам. */
    public List<EvaluationQuery> readQueries(Path path) throws IOException {
        var rows = readRows(path, EvaluationQuery.class);
        var ids = new HashSet<String>();

        for (var row : rows) {
            if (row == null || blank(row.id()) || row.text() == null) {
                throw new InvalidInputException("Запрос должен содержать непустой _id и строковый text: " + path);
            }

            if (!ids.add(row.id())) {
                throw new InvalidInputException("Повтор ID запроса: " + row.id());
            }
        }

        return rows;
    }

    /** Читает положительные связи, проверяя целые оценки, query-id и дубликаты. */
    public Map<String, List<RelevanceJudgment>> readJudgments(Path path, List<EvaluationQuery> queries)
            throws IOException {
        var queryIds = new HashSet<String>();
        queries.forEach(query -> queryIds.add(query.id()));
        var judgments = new HashMap<String, List<RelevanceJudgment>>();

        for (var row : readRows(path, RelevanceJudgment.class)) {
            if (
                row == null || blank(row.queryId()) || blank(row.documentId()) || row.score() == null || row.score() < 0
            ) {
                throw new InvalidInputException(
                        "Разметка должна содержать query-id, corpus-id и целый score >= 0: " + path
                );
            }

            if (!queryIds.contains(row.queryId())) {
                throw new InvalidInputException("Неизвестный query-id в разметке: " + row.queryId());
            }

            if (row.score() == 0) {
                continue;
            }
            var relations = judgments.computeIfAbsent(row.queryId(), ignored -> new ArrayList<>());

            if (relations.stream().anyMatch(previous -> previous.documentId().equals(row.documentId()))) {
                throw new InvalidInputException(
                        "Повтор связи запроса " + row.queryId() + " со статьёй " + row.documentId()
                );
            }
            relations.add(row);
        }

        judgments.values().forEach(rows -> rows.sort(Comparator.comparing(RelevanceJudgment::documentId)));

        return judgments;
    }

    private <T> List<T> readRows(Path path, Class<T> type) throws IOException {
        var rows = new ArrayList<T>();
        var parser = jsonMapper.readerFor(type).without(DeserializationFeature.ACCEPT_FLOAT_AS_INT);

        try (var reader = Files.newBufferedReader(path, StandardCharsets.UTF_8)) {
            String line;
            int number = 0;

            while ((line = reader.readLine()) != null) {
                number++;

                if (!line.isBlank()) {
                    try {
                        rows.add(parser.readValue(line));
                    } catch (IOException exception) {
                        throw new IOException(path + ", строка " + number + ": " + exception.getMessage(), exception);
                    }
                }
            }
        }

        return rows;
    }

    private static boolean blank(String value) {
        return value == null || value.isBlank();
    }
}
