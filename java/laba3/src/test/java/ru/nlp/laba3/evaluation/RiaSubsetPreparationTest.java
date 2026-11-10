package ru.nlp.laba3.evaluation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.corpus.reader.JsonlCorpusReader;
import ru.nlp.laba3.exception.CorpusReadException;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.evaluation.RelevanceJudgment;
import ru.nlp.laba3.model.evaluation.RiaSubsetManifest;
import ru.nlp.laba3.model.evaluation.SubsetParameters;

class RiaSubsetPreparationTest {

    @TempDir
    Path directory;

    @Test
    void retainsEveryRelevantArticleBeyondPrefixAndRoundTripsOriginalBodies() throws Exception {
        var corpus = archive(
                List.of(
                        new Document("bg", "Фон", "Фоновая статья экономики"),
                        new Document("copy", "Копия", "Фоновая статья экономики"),
                        new Document("empty", "", "и в на !"),
                        new Document("bg2", "Фон", "Исторические исследования"),
                        new Document("bg3", "Фон", "Медицинская конференция"),
                        new Document("bg4", "Фон", "Театральные спектакли"),
                        new Document("late1", "Ракеты, спутники", "Исходный текст, ракеты\nВторая строка \"цитата\""),
                        new Document("late2", "Матчи", "Футбол команды матчи"),
                        new Document("late3", "Песни", "Музыка концерты песни")
                )
        );
        var queries = rows(
                "queries.jsonl",
                List.of(query("q1", "Ракеты спутники"), query("q2", "Музыкальный концерт"))
        );
        var qrels = rows(
                "qrels.jsonl",
                List.of(relation("q1", "late1"), relation("q1", "late2"), relation("q2", "late3"))
        );
        var output = directory.resolve("коллекция с пробелом");
        var result = preparation().prepare(corpus, queries, qrels, output, new SubsetParameters(4, 2, 42));
        var documents = TestComponents.loader().read(output.resolve("corpus.csv"), 10).documents();
        assertThat(documents).extracting(Document::id).containsExactlyInAnyOrder("late1", "late2", "late3", "bg");
        assertThat(documents.stream().filter(doc -> doc.id().equals("late1")).findFirst().orElseThrow().text())
                .isEqualTo("Исходный текст, ракеты\nВторая строка \"цитата\"");
        assertThat(result.statistics().judgments()).isEqualTo(3);
        assertThat(result.statistics().relevantDocuments()).isEqualTo(3);
        assertThat(result.statistics().emptyCandidates()).isEqualTo(1);
        assertThat(result.statistics().duplicateBackgroundCandidates()).isEqualTo(1);
        var mapper = new JsonConfiguration().jsonMapper();
        var manifest = mapper.readValue(output.resolve("manifest.json").toFile(), RiaSubsetManifest.class);
        assertThat(manifest.documentIds()).containsExactlyElementsOf(documents.stream().map(Document::id).toList());

        for (var entry : manifest.outputSha256().entrySet()) {
            assertThat(FileChecksums.hash(output.resolve(entry.getKey()))).isEqualTo(entry.getValue());
        }

        for (String line : Files.readAllLines(output.resolve("qrels.jsonl"))) {
            var row = mapper.readValue(line, RelevanceJudgment.class);
            assertThat(manifest.queryIds()).contains(row.queryId());
            assertThat(manifest.documentIds()).contains(row.documentId());
        }
    }

    @Test
    void recordsExclusionsAndMatchesAndReproducesAllFilesByteForByte() throws Exception {
        var corpus = archive(
                List.of(
                        new Document("a", "Ёлки  РАКЕТЫ", "Космические аппараты исследуют планеты"),
                        new Document("b", "Музыка", "Музыка концерты песни"),
                        new Document("c", "Спорт", "Футбол команды матчи"),
                        new Document("empty", "Пустая статья", "и в на")
                )
        );
        var queries = rows(
                "queries.jsonl",
                List.of(
                        query("a", "елки ракеты"),
                        query("b", "Музыка концерты песни"),
                        query("copy", "МУЗЫКА концерты песни"),
                        query("c", "Спортивные соревнования"),
                        query("empty", "и в на !"),
                        query("missing", "Новости экономики"),
                        query("unusable", "Политические новости"),
                        query("unjudged", "Здоровье человека")
                )
        );
        var qrels = rows(
                "qrels.jsonl",
                List.of(
                        relation("a", "a"),
                        relation("b", "b"),
                        relation("copy", "b"),
                        relation("c", "c"),
                        relation("empty", "a"),
                        relation("missing", "absent"),
                        relation("unusable", "empty"),
                        new RelevanceJudgment("unjudged", "c", 0)
                )
        );
        var first = directory.resolve("first");
        var second = directory.resolve("second");
        var parameters = new SubsetParameters(3, 3, 42);
        var result = preparation().prepare(corpus, queries, qrels, first, parameters);
        preparation().prepare(corpus, queries, qrels, second, parameters);
        assertThat(result.statistics().excludedQueries()).containsEntry("empty_query", 1)
                .containsEntry("missing_or_empty_relevant_document", 2).containsEntry("duplicate_query", 1)
                .containsEntry("no_positive_judgment", 1);
        assertThat(result.statistics().queryTitleMatches()).isEqualTo(1);
        assertThat(result.statistics().queryBodyMatches()).isEqualTo(1);

        for (String name : List.of("manifest.json", "corpus.csv", "queries.jsonl", "qrels.jsonl")) {
            assertThat(Files.readAllBytes(first.resolve(name))).isEqualTo(Files.readAllBytes(second.resolve(name)));
        }
    }

    @Test
    void rejectsDuplicateRelevantBodiesWithoutRewritingJudgedIds() throws Exception {
        var corpus = archive(
                List.of(
                        new Document("a", "", "Ракеты спутники"),
                        new Document("alias", "", "Ракеты спутники"),
                        new Document("b", "", "Музыка песни")
                )
        );
        var queries = rows("queries.jsonl", List.of(query("q1", "Космос"), query("q2", "Музыка")));
        var qrels = rows("qrels.jsonl", List.of(relation("q1", "a"), relation("q1", "alias"), relation("q2", "b")));
        var output = directory.resolve("impossible");
        assertThatThrownBy(() -> preparation().prepare(corpus, queries, qrels, output, new SubsetParameters(2, 2, 42)))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("duplicate_relevant_body");
        assertThat(output).doesNotExist();
    }

    @Test
    void refusesInsufficientDocumentBudgetRatherThanDroppingRelevantRelations() throws Exception {
        var corpus = archive(
                List.of(
                        new Document("a", "", "Ракеты спутники"),
                        new Document("b", "", "Музыка песни"),
                        new Document("c", "", "Футбол матчи")
                )
        );
        var queries = rows("queries.jsonl", List.of(query("q", "Новости")));
        var qrels = rows("qrels.jsonl", List.of(relation("q", "a"), relation("q", "b"), relation("q", "c")));
        var output = directory.resolve("impossible");
        assertThatThrownBy(() -> preparation().prepare(corpus, queries, qrels, output, new SubsetParameters(2, 1, 42)))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("document_limit");
        assertThat(output).doesNotExist();
    }

    @Test
    void rejectsUnknownQueryAndDuplicateQueryIdsBeforeWritingOutput() throws Exception {
        var queries = rows("queries.jsonl", List.of(query("q", "Новости")));
        var qrels = rows("qrels.jsonl", List.of(relation("unknown", "a")));
        var output = directory.resolve("invalid");
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("absent.zip"),
                        queries,
                        qrels,
                        output,
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Неизвестный query-id");
        rows("queries.jsonl", List.of(query("q", "Новости"), query("q", "Спорт")));
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("absent.zip"),
                        queries,
                        qrels,
                        output,
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Повтор ID");
        assertThat(output).doesNotExist();
    }

    @Test
    void refusesOverwriteBeforeReadingSources() throws Exception {
        var output = Files.createDirectory(directory.resolve("existing"));
        Files.writeString(output.resolve("sentinel"), "keep");
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("absent.zip"),
                        directory.resolve("q"),
                        directory.resolve("r"),
                        output,
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("уже существует");
        assertThat(Files.readString(output.resolve("sentinel"))).isEqualTo("keep");
    }

    @Test
    void countsUnselectedEligibleQueriesWithoutRemovingTheirRelevantArticlesFromBackground() throws Exception {
        var corpus = archive(
                List.of(
                        new Document("a", "", "Ракеты спутники"),
                        new Document("b", "", "Футбол матчи"),
                        new Document("c", "", "Музыка песни")
                )
        );
        var queries = rows(
                "queries.jsonl",
                List.of(query("q1", "Космос"), query("q2", "Спорт"), query("q3", "Концерты"))
        );
        var qrels = rows("qrels.jsonl", List.of(relation("q1", "a"), relation("q2", "b"), relation("q3", "c")));
        var output = directory.resolve("limited");
        var result = preparation().prepare(corpus, queries, qrels, output, new SubsetParameters(3, 1, 42));
        assertThat(result.statistics().excludedQueries()).containsOnlyKeys("query_limit")
                .containsEntry("query_limit", 2);
        assertThat(result.statistics().relevantDocuments()).isEqualTo(1);
        assertThat(TestComponents.loader().read(output.resolve("corpus.csv"), 3).documents()).extracting(Document::id)
                .containsExactlyInAnyOrder("a", "b", "c");
    }

    @ParameterizedTest
    @ValueSource(strings = {"null", "{}", "{\"_id\":\"\",\"text\":\"Новости\"}", "{\"_id\":\"q\"}"})
    void rejectsIncompleteQueriesBeforeReadingCorpus(String line) throws Exception {
        var queries = Files.writeString(directory.resolve("queries.jsonl"), line + "\n");
        var output = directory.resolve("invalid");
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("missing.zip"),
                        queries,
                        directory.resolve("missing-qrels.jsonl"),
                        output,
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Запрос должен содержать");
        assertThat(output).doesNotExist();
    }

    @ParameterizedTest
    @ValueSource(
            strings = {"null", "{\"query-id\":\"q\",\"corpus-id\":\"a\"}",
                    "{\"query-id\":\"q\",\"corpus-id\":\"a\",\"score\":-1}",
                    "{\"query-id\":\"\",\"corpus-id\":\"a\",\"score\":1}",
                    "{\"query-id\":\"q\",\"corpus-id\":\"\",\"score\":1}"}
    )
    void rejectsInvalidJudgmentsWithoutCreatingOutput(String line) throws Exception {
        var queries = rows("queries.jsonl", List.of(query("q", "Новости")));
        var qrels = Files.writeString(directory.resolve("qrels.jsonl"), line + "\n");
        var output = directory.resolve("invalid");
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("missing.zip"),
                        queries,
                        qrels,
                        output,
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Разметка должна содержать");
        assertThat(output).doesNotExist();
    }

    @Test
    void malformedJsonKeepsSourcePathLineAndCause() throws Exception {
        var queries = Files.writeString(directory.resolve("queries.jsonl"), "\nnot json\n");
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("missing.zip"),
                        queries,
                        directory.resolve("missing-qrels.jsonl"),
                        directory.resolve("invalid"),
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(CorpusReadException.class).hasCauseInstanceOf(java.io.IOException.class)
                .hasMessageContaining("queries.jsonl").hasMessageContaining("строка 2");
    }

    @Test
    void rejectsFractionalScoresInsteadOfTruncatingThem() throws Exception {
        var queries = rows("queries.jsonl", List.of(query("q", "Новости")));
        var qrels = Files.writeString(
                directory.resolve("qrels.jsonl"),
                "{\"query-id\":\"q\",\"corpus-id\":\"a\",\"score\":1.5}\n"
        );
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("missing.zip"),
                        queries,
                        qrels,
                        directory.resolve("invalid"),
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(CorpusReadException.class).hasCauseInstanceOf(java.io.IOException.class)
                .hasMessageContaining("qrels.jsonl");
    }

    @Test
    void rejectsRepeatedJudgmentAndExcessQueryCountBeforeScanningCorpus() throws Exception {
        var queries = rows("queries.jsonl", List.of(query("q", "Новости")));
        var qrels = rows("qrels.jsonl", List.of(relation("q", "a"), relation("q", "a")));
        var output = directory.resolve("invalid");
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("missing.zip"),
                        queries,
                        qrels,
                        output,
                        new SubsetParameters(2, 1, 42)
                )
        )
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Повтор связи");
        assertThatThrownBy(
                () -> preparation().prepare(
                        directory.resolve("missing.zip"),
                        queries,
                        qrels,
                        output,
                        new SubsetParameters(2, 2, 42)
                )
        )
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("больше запросов");
        assertThat(output).doesNotExist();
    }

    @Test
    void refusesUnderfilledCollectionEvenWhenQueryHasAllRelevantDocuments() throws Exception {
        var corpus = archive(List.of(new Document("a", "", "Ракеты спутники"), new Document("b", "", "Футбол матчи")));
        var queries = rows("queries.jsonl", List.of(query("q", "Космос")));
        var qrels = rows("qrels.jsonl", List.of(relation("q", "a")));
        var output = directory.resolve("underfilled");
        assertThatThrownBy(() -> preparation().prepare(corpus, queries, qrels, output, new SubsetParameters(3, 1, 42)))
                .isInstanceOf(InvalidInputException.class)
                .hasMessageContaining("Недостаточно уникальных непустых статей");
        assertThat(output).doesNotExist();
    }

    @Test
    void streamingReaderStopsBeforeMalformedTailAndIgnoresZipMetadata() throws Exception {
        var path = rows(
                "prefix.jsonl",
                List.of(new java.util.LinkedHashMap<>(java.util.Map.of("_id", "a", "title", "", "text", "Ракеты")))
        );
        Files.writeString(path, "not json", StandardCharsets.UTF_8, java.nio.file.StandardOpenOption.APPEND);
        var reader = new JsonlCorpusReader(new JsonConfiguration().jsonMapper());
        var documents = new java.util.ArrayList<Document>();
        assertThat(reader.scan(path, doc -> {
            documents.add(doc);

            return false;
        })).isEqualTo(1);
        assertThat(documents).extracting(Document::id).containsExactly("a");
        assertThat(reader.scan(archive(List.of(new Document("a", "", "Ракеты"))), ignored -> true)).isEqualTo(1);
    }

    private RiaSubsetPreparation preparation() throws Exception {
        var configuration = new JsonConfiguration();

        return new RiaSubsetPreparation(
                configuration.jsonMapper(),
                configuration.csvMapper(),
                TestComponents.corpusReader(),
                TestComponents.preprocessor(),
                new EvaluationDataReader(configuration.jsonMapper())
        );
    }

    private Path archive(List<Document> documents) throws Exception {
        var path = directory.resolve("corpus.zip");
        var mapper = new JsonConfiguration().jsonMapper();

        try (var zip = new ZipOutputStream(Files.newOutputStream(path), StandardCharsets.UTF_8)) {
            zip.putNextEntry(new ZipEntry("__MACOSX/._corpus.jsonl"));
            zip.write("not json".getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
            zip.putNextEntry(new ZipEntry("corpus.jsonl"));

            for (var document : documents) {
                zip.write(
                        (mapper.writeValueAsString(
                                java.util.Map
                                        .of("_id", document.id(), "title", document.title(), "text", document.text())
                        ) + "\n")
                                .getBytes(StandardCharsets.UTF_8)
                );
            }

            zip.closeEntry();
        }

        return path;
    }

    private Path rows(String name, List<?> values) throws Exception {
        var path = directory.resolve(name);
        var mapper = new JsonConfiguration().jsonMapper();

        try (var writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (var value : values) {
                writer.write(mapper.writeValueAsString(value) + "\n");
            }
        }

        return path;
    }

    private static EvaluationQuery query(String id, String text) {
        return new EvaluationQuery(id, text);
    }

    private static RelevanceJudgment relation(String query, String document) {
        return new RelevanceJudgment(query, document, 1);
    }
}
