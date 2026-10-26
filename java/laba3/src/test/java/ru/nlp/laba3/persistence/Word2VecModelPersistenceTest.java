package ru.nlp.laba3.persistence;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.fasterxml.jackson.databind.node.ObjectNode;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.exception.ModelExportException;
import ru.nlp.laba3.exception.ModelImportException;
import ru.nlp.laba3.model.word2vec.Word2VecParameters;
import ru.nlp.laba3.preprocessing.TextNormalizationChain;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.preprocessing.rules.LowercaseRule;
import ru.nlp.laba3.preprocessing.rules.RussianYoRule;
import ru.nlp.laba3.preprocessing.rules.UnicodeNfcRule;
import ru.nlp.laba3.runtime.NativeRuntime;

class Word2VecModelPersistenceTest {

    @TempDir
    Path directory;

    @Test
    void persistedModelRebuildsIndexAndHandlesOovAndEmptyQueries() throws Exception {
        var input = directory.resolve("texts");
        Files.createDirectories(input);
        Files.writeString(input.resolve("space.txt"), "Космос ракеты спутники космос ракеты спутники");
        Files.writeString(input.resolve("sport.txt"), "Футбол матчи команды футбол матчи команды");
        Files.writeString(input.resolve("music.txt"), "Музыка концерты песни музыка концерты песни");
        var backend = TestComponents.word2VecBackend();
        assertThatThrownBy(
                () -> backend.train(
                        input,
                        directory.resolve("invalid-model"),
                        3,
                        new Word2VecParameters(8, 2, 100, 3, 42, 1)
                )
        )
                .hasMessageContaining("минимум 2 разных слова");
        var output = directory.resolve("model");
        backend.train(input, output, 3, new Word2VecParameters(8, 2, 1, 3, 42, 1));
        String query = "Космос ракеты спутники космос ракеты спутники";
        var first = backend.open(output).search(query, 3);
        var second = backend.open(output).search(query, 3);
        assertThat(first.status()).isEqualTo("ok");
        assertThat(first.hits().getFirst().id()).isEqualTo("space.txt");
        assertThat(first.hits().getFirst().cosine()).isCloseTo(1, within(1e-6));
        assertThat(second).isEqualTo(first);
        var mapper = new JsonConfiguration().jsonMapper();
        var manifest = (ObjectNode) mapper.readTree(output.resolve("manifest.json").toFile());
        assertThat(manifest.has("formatVersion")).isFalse();
        assertThat(manifest.has("corpus")).isFalse();
        assertThat(manifest.has("idf")).isFalse();
        var versionKeys = new java.util.ArrayList<String>();
        manifest.path("versions").fieldNames().forEachRemaining(versionKeys::add);
        assertThat(versionKeys.subList(versionKeys.size() - 6, versionKeys.size()))
                .containsExactly("java", "os", "arch", "nd4jBackend", "nd4jOmpMaxThreads", "jackson");
        assertThat(manifest.path("documents").asInt()).isEqualTo(3);
        assertThat(Files.size(output.resolve("data.json.gz"))).isPositive();
        var store = new Word2VecModelStore(mapper, TestComponents.preprocessor());
        var bundle = store.load(output);
        var index = backend.open(output);
        var blocker = directory.resolve("blocked");
        Files.writeString(blocker, "file");
        assertThatThrownBy(() -> store.save(blocker.resolve("model"), bundle))
                .isInstanceOf(ModelExportException.class).hasCauseInstanceOf(IOException.class);

        try (var executor = Executors.newFixedThreadPool(3)) {
            var blocked = NativeRuntime.withLock(() -> {
                var started = new CountDownLatch(1);
                var search = executor.submit(() -> {
                    started.countDown();

                    return index.search(query, 3);
                });
                assertThatThrownBy(() -> {
                    assertThat(started.await(5, TimeUnit.SECONDS)).isTrue();
                    search.get(100, TimeUnit.MILLISECONDS);
                })
                        .isInstanceOf(TimeoutException.class);

                return search;
            });
            assertThat(blocked.get(10, TimeUnit.SECONDS)).isEqualTo(first);
            var searches = java.util.stream.IntStream.range(0, 12)
                    .mapToObj(i -> executor.submit(() -> index.search(query, 3))).toList();

            for (var search : searches) {
                assertThat(search.get(10, TimeUnit.SECONDS)).isEqualTo(first);
            }
        }

        byte[] originalData = Files.readAllBytes(output.resolve("data.json.gz"));
        Files.writeString(output.resolve("data.json.gz"), "damaged", StandardOpenOption.APPEND);
        assertThatThrownBy(() -> store.load(output)).isInstanceOf(ModelImportException.class)
                .hasMessageContaining("Повреждены данные");
        Files.write(output.resolve("data.json.gz"), originalData);
        var changedChain = TextNormalizationChain.builder()
                .rule(new RussianYoRule())
                .rule(new LowercaseRule())
                .rule(new UnicodeNfcRule())
                .build();
        var changedPreprocessor = new TextPreprocessor(changedChain);
        var incompatibleStore = new Word2VecModelStore(
                new JsonConfiguration().jsonMapper(),
                changedPreprocessor
        );
        assertThatThrownBy(() -> incompatibleStore.load(output)).hasMessageContaining("профиль нормализации");
        assertThat(backend.open(output).search("абракадабравыв", 3).status()).isEqualTo("oov");
        assertThat(backend.open(output).search("и в на ?!", 3).status())
                .isEqualTo("empty_after_normalization");
        assertThatThrownBy(
                () -> backend.train(input, output, 3, new Word2VecParameters(8, 2, 1, 3, 42, 1))
        )
                .isInstanceOf(InvalidInputException.class)
                .hasMessageContaining("уже существует");
        Files.writeString(output.resolve("vectors.txt"), "damaged", StandardOpenOption.APPEND);
        assertThatThrownBy(() -> backend.open(output).search(query, 3))
                .isInstanceOf(ModelImportException.class)
                .hasMessageContaining("Повреждены веса");
    }

    @ParameterizedTest
    @ValueSource(strings = {"parameters", "documents", "vocabularySize", "dimensions"})
    void rejectsManifestThatDisagreesWithUnchangedPayload(String field) throws Exception {
        var input = Files.createDirectory(directory.resolve("texts"));
        Files.writeString(input.resolve("space.txt"), "Космос ракеты спутники");
        Files.writeString(input.resolve("sport.txt"), "Футбол команды матчи");
        var output = directory.resolve("model");
        TestComponents.word2VecBackend().train(input, output, 2, new Word2VecParameters(4, 1, 1, 2, 42, 1));
        var mapper = new JsonConfiguration().jsonMapper();
        var manifest = (ObjectNode) mapper.readTree(output.resolve("manifest.json").toFile());
        String expected;

        switch (field) {
            case "parameters" -> {
                manifest.putNull("parameters");
                expected = "отсутствуют параметры";
            }
            case "dimensions" -> {
                ((ObjectNode) manifest.path("parameters")).put("dimensions", 5);
                expected = "не соответствует весам";
            }
            case "vocabularySize" -> {
                manifest.put("vocabularySize", manifest.path("vocabularySize").asInt() + 1);
                expected = "не соответствует весам";
            }
            default -> {
                manifest.put("documents", 3);
                expected = "не соответствует данным";
            }
        }

        mapper.writeValue(output.resolve("manifest.json").toFile(), manifest);
        var store = new Word2VecModelStore(mapper, TestComponents.preprocessor());
        assertThatThrownBy(() -> store.load(output)).isInstanceOf(ModelImportException.class)
                .hasMessageContaining(expected);
    }

    @Test
    void importErrorsRetainFileCauseAndContext() throws Exception {
        var store = new Word2VecModelStore(new JsonConfiguration().jsonMapper(), TestComponents.preprocessor());
        var missing = directory.resolve("missing-model");
        assertThatThrownBy(() -> store.load(missing)).isInstanceOf(ModelImportException.class)
                .hasMessageContaining(missing.toString()).hasCauseInstanceOf(IOException.class);
    }
}
