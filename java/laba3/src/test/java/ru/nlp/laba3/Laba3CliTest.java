package ru.nlp.laba3;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.slf4j.Logger;
import org.springframework.boot.logging.LogLevel;
import org.springframework.boot.logging.LoggingSystem;

import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.model.analysis.Algorithm;

class Laba3CliTest {

    @TempDir
    Path directory;

    @Test
    void normalizationAndHelpAreQuietAndPreserveRussianText() {
        var normalization = execute("--text", "Ёлками и машины НЕ были ни 12,5");
        assertThat(normalization.exitCode()).isZero();
        assertThat(normalization.stdout().lines()).containsExactly("елк", "машин", "не", "ни", "12,5");
        assertThat(normalization.stderr()).isEmpty();

        for (var args : List.of(new String[0], new String[]{"--help"}, new String[]{"prepare-ria", "--help"})) {
            var help = execute(args);
            assertThat(help.exitCode()).isZero();
            assertThat(help.stdout()).contains("Usage:");
            assertThat(help.stderr()).isEmpty();
        }
    }

    @ParameterizedTest
    @EnumSource(Algorithm.class)
    void trainsAndSearchesThroughSpringWithIsolatedJsonAndOptInLogs(Algorithm algorithm) throws Exception {
        var input = directory.resolve("русские документы.csv");
        Files.writeString(input, """
                id,title,text
                space-a,Космос,Космос ракеты спутники орбита
                space-b,Космос,Космос ракеты спутники корабль
                sport-a,Спорт,Футбол команды матчи чемпионат
                sport-b,Спорт,Футбол команды матчи стадион
                music-a,Музыка,Музыка концерты песни оркестр
                music-b,Музыка,Музыка концерты песни гитара
                """);
        var output = directory.resolve("модель " + algorithm);
        var args = new ArrayList<>(
                List.of(
                        "train",
                        "--verbose",
                        "--algorithm",
                        algorithm.name(),
                        "--input",
                        input.toString(),
                        "--output",
                        output.toString(),
                        "--limit",
                        "6",
                        "--dimensions",
                        "3"
                )
        );

        if (algorithm == Algorithm.word2vec) {
            args.addAll(List.of("--epochs", "1", "--min-count", "1"));
        } else {
            args.addAll(List.of("--max-features", "100"));
        }
        var trained = execute(args.toArray(String[]::new));
        assertThat(trained.exitCode()).isZero();
        var mapper = new JsonConfiguration().jsonMapper();
        assertThat(mapper.readTree(trained.stdout()).path("documents").asInt()).isEqualTo(6);
        assertThat(trained.stderr()).contains("[" + algorithm.displayName() + "]");
        var query = List.of(
                "search",
                "--algorithm",
                algorithm.name(),
                "--model",
                output.toString(),
                "--query",
                "Ракеты спутники",
                "--top",
                "3"
        );
        var quiet = execute(query.toArray(String[]::new));
        assertThat(quiet.exitCode()).isZero();
        assertThat(quiet.stderr()).isEmpty();
        var result = mapper.readTree(quiet.stdout());
        assertThat(result.path("status").asText()).isEqualTo("ok");
        assertThat(result.path("knownTokens").asInt()).isEqualTo(2);
        assertThat(result.path("hits").size()).isEqualTo(3);
        var verboseArgs = new ArrayList<>(List.of("-v"));
        verboseArgs.addAll(query);
        var verbose = execute(verboseArgs.toArray(String[]::new));
        assertThat(verbose.exitCode()).isZero();
        assertThat(mapper.readTree(verbose.stdout())).isEqualTo(result);
        assertThat(verbose.stderr()).contains("[" + algorithm.displayName() + "]");
    }

    @ParameterizedTest
    @CsvSource({"lsa,--epochs", "lsa,--min-count", "lsa,--window", "lsa,--seed", "word2vec,--max-features"})
    void rejectsOptionsOfAnotherAlgorithmBeforeReadingInput(String algorithm, String option) {
        var result = execute(
                "train",
                "--algorithm",
                algorithm,
                "--input",
                directory.resolve("missing.csv").toString(),
                "--output",
                directory.resolve("model").toString(),
                option,
                "2"
        );
        assertThat(result.exitCode()).isEqualTo(1);
        assertThat(result.stdout()).isEmpty();
        assertThat(result.stderr()).startsWith("Ошибка:").contains("только к");
        assertThat(directory.resolve("model")).doesNotExist();
    }

    @ParameterizedTest
    @CsvSource(
        {"lsa,--dimensions,0", "lsa,--max-features,1", "lsa,--workers,0", "word2vec,--dimensions,1",
                "word2vec,--epochs,0", "word2vec,--min-count,0", "word2vec,--window,0", "word2vec,--workers,0"}
    )
    void rejectsInvalidTrainingBoundsBeforeReadingInput(String algorithm, String option, String value) {
        var result = execute(
                "train",
                "--algorithm",
                algorithm,
                "--input",
                directory.resolve("missing.csv").toString(),
                "--output",
                directory.resolve("model").toString(),
                option,
                value
        );
        assertThat(result.exitCode()).isEqualTo(1);
        assertThat(result.stdout()).isEmpty();
        assertThat(result.stderr()).startsWith("Ошибка:");
        assertThat(result.stderr()).doesNotContain("Не удалось прочитать корпус");
        assertThat(directory.resolve("model")).doesNotExist();
    }

    @Test
    void preparesRiaSubsetThroughCliWithoutLogsOrTraining() throws Exception {
        var input = directory.resolve("corpus.jsonl");
        Files.writeString(input, """
                {"_id":"a","title":"Космос","text":"Ракеты спутники"}
                {"_id":"b","title":"Спорт","text":"Футбол команды"}
                {"_id":"c","title":"Музыка","text":"Концерты песни"}
                """);
        var queries = Files.writeString(directory.resolve("queries.jsonl"), "{\"_id\":\"q\",\"text\":\"Космос\"}\n");
        var qrels = Files.writeString(
                directory.resolve("qrels.jsonl"),
                "{\"query-id\":\"q\",\"corpus-id\":\"a\",\"score\":1}\n"
        );
        var output = directory.resolve("выборка РИА");
        var result = execute(
                "prepare-ria",
                "--corpus",
                input.toString(),
                "--queries",
                queries.toString(),
                "--qrels",
                qrels.toString(),
                "--output",
                output.toString(),
                "--documents",
                "3",
                "--query-count",
                "1",
                "--seed",
                "42"
        );
        assertThat(result.exitCode()).isZero();
        assertThat(result.stderr()).isEmpty();
        var summary = new JsonConfiguration().jsonMapper().readTree(result.stdout());
        assertThat(summary.path("statistics").path("documents").asInt()).isEqualTo(3);
        assertThat(summary.path("statistics").path("queries").asInt()).isEqualTo(1);
        assertThat(output.resolve("manifest.json")).exists();
        assertThat(output.resolve("qrels.jsonl")).hasContent("{\"query-id\":\"q\",\"corpus-id\":\"a\",\"score\":1}\n");
    }

    @ParameterizedTest
    @CsvSource({"--documents,1", "--query-count,0"})
    void rejectsSubsetBoundsBeforeReadingSources(String option, String value) {
        var output = directory.resolve("invalid subset");
        var result = execute(
                "prepare-ria",
                "--corpus",
                directory.resolve("missing.zip").toString(),
                "--queries",
                directory.resolve("queries.jsonl").toString(),
                "--qrels",
                directory.resolve("qrels.jsonl").toString(),
                "--output",
                output.toString(),
                option,
                value
        );
        assertThat(result.exitCode()).isEqualTo(1);
        assertThat(result.stdout()).isEmpty();
        assertThat(result.stderr()).startsWith("Ошибка:").contains("не меньше 2 документов и 1 запроса");
        assertThat(output).doesNotExist();
    }

    @Test
    void reportsParserErrorsAndExecutionErrorsWithDifferentExitCodes() {
        var invalid = execute("--unknown-option");
        assertThat(invalid.exitCode()).isEqualTo(2);
        assertThat(invalid.stdout()).isEmpty();
        assertThat(invalid.stderr()).contains("Unknown option");
        var missing = execute(
                "search",
                "--algorithm",
                "lsa",
                "--model",
                directory.resolve("missing").toString(),
                "--query",
                "Ракеты"
        );
        assertThat(missing.exitCode()).isEqualTo(1);
        assertThat(missing.stdout()).isEmpty();
        assertThat(missing.stderr()).startsWith("Ошибка:").contains("LSA");
    }

    @Test
    void evaluatesBothSavedModelsWithoutFilteringOovOrOverwritingResults() throws Exception {
        var mapper = new JsonConfiguration().jsonMapper();
        var source = directory.resolve("corpus.jsonl");
        var queries = directory.resolve("queries.jsonl");
        var qrels = directory.resolve("qrels.jsonl");
        Files.writeString(source, """
                {"_id":"space-a","title":"Космос","text":"Космос ракеты спутники орбита"}
                {"_id":"space-b","title":"Космос","text":"Космос ракеты спутники корабль"}
                {"_id":"sport-a","title":"Спорт","text":"Футбол команды матчи чемпионат"}
                {"_id":"sport-b","title":"Спорт","text":"Футбол команды матчи стадион"}
                {"_id":"music-a","title":"Музыка","text":"Музыка концерты песни оркестр"}
                {"_id":"music-b","title":"Музыка","text":"Музыка концерты песни гитара"}
                """);
        Files.writeString(queries, """
                {"_id":"q1","text":"Ракеты спутники"}
                {"_id":"q2","text":"Футбол команды"}
                {"_id":"q3","text":"Музыка концерты"}
                {"_id":"q4","text":"абракадабравыв"}
                """);
        Files.writeString(qrels, """
                {"query-id":"q1","corpus-id":"space-a","score":1}
                {"query-id":"q2","corpus-id":"sport-a","score":1}
                {"query-id":"q3","corpus-id":"music-a","score":1}
                {"query-id":"q4","corpus-id":"space-a","score":1}
                """);
        var collection = directory.resolve("общая коллекция");
        assertThat(
                execute(
                        "prepare-ria",
                        "--corpus",
                        source.toString(),
                        "--queries",
                        queries.toString(),
                        "--qrels",
                        qrels.toString(),
                        "--output",
                        collection.toString(),
                        "--documents",
                        "6",
                        "--query-count",
                        "4"
                ).exitCode()
        ).isZero();
        var models = new java.util.EnumMap<Algorithm, Path>(Algorithm.class);

        for (var algorithm : Algorithm.values()) {
            var model = directory.resolve("оценочная модель " + algorithm);
            var args = new ArrayList<>(
                    List.of(
                            "train",
                            "--algorithm",
                            algorithm.name(),
                            "--input",
                            collection.resolve("corpus.csv").toString(),
                            "--output",
                            model.toString(),
                            "--limit",
                            "6",
                            "--dimensions",
                            "3"
                    )
            );

            if (algorithm == Algorithm.word2vec) {
                args.addAll(List.of("--epochs", "1", "--min-count", "1"));
            }
            assertThat(execute(args.toArray(String[]::new)).exitCode()).isZero();
            models.put(algorithm, model);
        }

        for (var verbose : List.of(false, true)) {
            var output = directory.resolve("результат " + verbose);
            var args = new ArrayList<>(
                    List.of(
                            "evaluate",
                            "--collection",
                            collection.toString(),
                            "--lsa-model",
                            models.get(Algorithm.lsa).toString(),
                            "--word2vec-model",
                            models.get(Algorithm.word2vec).toString(),
                            "--output",
                            output.toString()
                    )
            );

            if (verbose) {
                args.add("--verbose");
            }
            var result = execute(args.toArray(String[]::new));
            assertThat(result.exitCode()).isZero();

            if (verbose) {
                assertThat(result.stderr()).contains("[LSA]", "[Word2Vec]", "[Оценка]");
            } else {
                assertThat(result.stderr()).isEmpty();
            }
            var summary = mapper.readTree(result.stdout());
            assertThat(summary.path("queries").asInt()).isEqualTo(4);
            var reportFile = output.resolve("report.json");
            byte[] bytes = Files.readAllBytes(reportFile);
            var report = mapper.readTree(bytes);
            assertThat(report.path("models").size()).isEqualTo(2);

            for (var evaluated : report.path("models")) {
                assertThat(evaluated.path("metrics").path("queries").asInt()).isEqualTo(4);
                assertThat(evaluated.path("results").size()).isEqualTo(4);
                assertThat(evaluated.path("metrics").path("statuses").path("oov").asInt()).isEqualTo(1);
                assertThat(evaluated.path("metrics").path("recallAt5").asDouble()).isLessThanOrEqualTo(0.75);
                assertThat(evaluated.path("metadata").path("parameters").path("dimensions").asInt()).isEqualTo(3);
                assertThat(evaluated.path("metadata").path("versions").has("java")).isTrue();
                assertThat(evaluated.path("modelSha256").has("manifest.json")).isTrue();
            }

            assertThat(execute(args.toArray(String[]::new)).exitCode()).isEqualTo(1);
            assertThat(Files.readAllBytes(reportFile)).isEqualTo(bytes);
        }

        var small = directory.resolve("неполная модель");
        assertThat(
                execute(
                        "train",
                        "--algorithm",
                        "lsa",
                        "--input",
                        collection.resolve("corpus.csv").toString(),
                        "--output",
                        small.toString(),
                        "--limit",
                        "3",
                        "--dimensions",
                        "2"
                ).exitCode()
        ).isZero();
        var rejectedOutput = directory.resolve("отказ");
        var mismatch = execute(
                "evaluate",
                "--collection",
                collection.toString(),
                "--lsa-model",
                small.toString(),
                "--word2vec-model",
                models.get(Algorithm.word2vec).toString(),
                "--output",
                rejectedOutput.toString()
        );
        assertThat(mismatch.exitCode()).isEqualTo(1);
        assertThat(mismatch.stderr()).contains("другом составе");
        assertThat(Files.exists(rejectedOutput)).isFalse();
        assertThat(execute("evaluate").exitCode()).isEqualTo(2);
    }

    private CliResult execute(String... args) {
        var previousOut = System.out;
        var previousErr = System.err;
        var logging = LoggingSystem.get(Laba3Application.class.getClassLoader());
        var configuration = logging.getLoggerConfiguration(Logger.ROOT_LOGGER_NAME);
        var previousLevel = configuration == null ? LogLevel.INFO : configuration.getEffectiveLevel();
        var stdout = new ByteArrayOutputStream();
        var stderr = new ByteArrayOutputStream();

        try (
                var out = new PrintStream(stdout, true, StandardCharsets.UTF_8);
                var err = new PrintStream(stderr, true, StandardCharsets.UTF_8)
        ) {
            System.setOut(out);
            System.setErr(err);
            int code = Laba3Application.execute(args);

            return new CliResult(
                    code,
                    stdout.toString(StandardCharsets.UTF_8),
                    stderr.toString(StandardCharsets.UTF_8)
            );
        } finally {
            System.setOut(previousOut);
            System.setErr(previousErr);
            logging.setLogLevel(Logger.ROOT_LOGGER_NAME, previousLevel);
        }
    }

    private record CliResult(
            /** Код завершения команды. */
            int exitCode,
            /** UTF-8 результат команды. */
            String stdout,
            /** UTF-8 диагностика команды. */
            String stderr
    ) {}
}
