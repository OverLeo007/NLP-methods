package ru.nlp.laba3.evaluation;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.csv.CsvMapper;
import com.fasterxml.jackson.dataformat.csv.CsvSchema;

import ru.nlp.laba3.corpus.reader.CorpusReader;
import ru.nlp.laba3.exception.CorpusReadException;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.evaluation.RiaSubsetManifest;
import ru.nlp.laba3.model.evaluation.RiaSubsetSummary;
import ru.nlp.laba3.model.evaluation.SubsetParameters;
import ru.nlp.laba3.preprocessing.TextPreprocessor;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Готовит одну коллекцию для обоих анализаторов без обучения на запросах и разметке. */
@Component
@RequiredArgsConstructor
@Slf4j
public class RiaSubsetPreparation {

    private final ObjectMapper jsonMapper;

    private final CsvMapper csvMapper;

    private final CorpusReader corpusReader;

    private final TextPreprocessor preprocessor;

    private final EvaluationDataReader dataReader;

    /** Выбирает статьи и запросы, сохраняет данные и манифест в новый каталог. */
    public RiaSubsetSummary prepare(Path corpus, Path queries, Path qrels, Path output, SubsetParameters parameters) {
        if (Files.exists(output)) {
            throw new InvalidInputException("Каталог выборки уже существует: " + output);
        }
        long started = System.nanoTime();

        try {
            var queryRows = dataReader.readQueries(queries);

            if (queryRows.size() < parameters.queries()) {
                throw new InvalidInputException(
                        "Запрошено больше запросов, чем имеется в источнике: " + queryRows.size()
                );
            }
            var judgments = dataReader.readJudgments(qrels, queryRows);
            var targetIds = new HashSet<String>();
            judgments.values().forEach(rows -> rows.forEach(row -> targetIds.add(row.documentId())));
            log.info(
                    "[РИА] Подготовка коллекции: запросов={}, размеченных статей={}, seed={}",
                    queryRows.size(),
                    targetIds.size(),
                    parameters.seed()
            );
            var selector = new RiaSubsetSelector(parameters, preprocessor, targetIds);
            corpusReader.scan(corpus, selector::accept);
            var selection = selector.select(queryRows, judgments);
            log.info(
                    "[РИА] Отобрано запросов={}, статей={}, релевантных={}; проверка SHA-256 источников",
                    selection.queries().size(),
                    selection.documents().size(),
                    selection.statistics().relevantDocuments()
            );
            var sourceHashes = Map.of(
                    "corpus",
                    FileChecksums.hash(corpus),
                    "queries",
                    FileChecksums.hash(queries),
                    "qrels",
                    FileChecksums.hash(qrels)
            );
            write(output, parameters, selection, sourceHashes);
            double seconds = (System.nanoTime() - started) / 1_000_000_000.0;
            log.info("[РИА] Коллекция сохранена: {}, время={} с", output, seconds);

            return new RiaSubsetSummary(output.toString(), selection.statistics(), seconds);
        } catch (IOException exception) {
            throw new CorpusReadException("Не удалось подготовить выборку РИА: " + exception.getMessage(), exception);
        }
    }

    private void write(
            Path output,
            SubsetParameters parameters,
            RiaSubsetSelector.Result selection,
            Map<String, String> sourceHashes
    ) throws IOException {
        Files.createDirectories(output);
        var schema = CsvSchema.builder()
                .addColumn("id")
                .addColumn("title")
                .addColumn("text")
                .setUseHeader(true)
                .setLineSeparator("\n")
                .build();

        try (
                var writer = Files.newBufferedWriter(output.resolve("corpus.csv"), StandardCharsets.UTF_8);
                var rows = csvMapper.writerFor(Document.class).with(schema).writeValues(writer)
        ) {
            rows.writeAll(selection.documents());
        }

        writeRows(output.resolve("queries.jsonl"), selection.queries());
        writeRows(output.resolve("qrels.jsonl"), selection.judgments());
        var outputHashes = Map.of(
                "corpus.csv",
                FileChecksums.hash(output.resolve("corpus.csv")),
                "queries.jsonl",
                FileChecksums.hash(output.resolve("queries.jsonl")),
                "qrels.jsonl",
                FileChecksums.hash(output.resolve("qrels.jsonl"))
        );
        var manifest = RiaSubsetManifest.builder()
                .protocol("ria-known-collection-held-out-queries-v1")
                .parameters(parameters)
                .preprocessing(preprocessor.profile())
                .stopwordsSha256(preprocessor.resourceHash())
                .javaVersion(System.getProperty("java.version"))
                .sourceSha256(sourceHashes)
                .outputSha256(outputHashes)
                .queryIds(selection.queries().stream().map(EvaluationQuery::id).toList())
                .documentIds(selection.documents().stream().map(Document::id).toList())
                .statistics(selection.statistics())
                .build();
        Files.writeString(
                output.resolve("manifest.json"),
                jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(manifest) + "\n",
                StandardCharsets.UTF_8
        );
    }

    private void writeRows(Path path, List<?> rows) throws IOException {
        try (var writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (var row : rows) {
                writer.write(jsonMapper.writeValueAsString(row));
                writer.write('\n');
            }
        }
    }

}
