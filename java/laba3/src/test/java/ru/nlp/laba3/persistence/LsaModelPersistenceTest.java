package ru.nlp.laba3.persistence;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.fasterxml.jackson.databind.node.ObjectNode;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.backend.LsaBackend;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.corpus.CorpusPreparation;
import ru.nlp.laba3.exception.ModelImportException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.corpus.Corpus;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.lsa.LsaManifest;
import ru.nlp.laba3.model.lsa.LsaModel;
import ru.nlp.laba3.model.lsa.LsaModelData;
import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.model.storage.ModelBundle;
import ru.nlp.laba3.runtime.RuntimeMetadata;
import ru.nlp.laba3.search.SearchSession;
import ru.nlp.laba3.service.LsaService;
import ru.nlp.laba3.training.LsaTrainer;
import ru.nlp.laba3.vectorization.LsaVectorizer;

class LsaModelPersistenceTest {

    @TempDir
    Path directory;

    @Test
    void savedModelSearchesUserDataAndRejectsDamageAndOverwrite() throws Exception {
        var mapper = new JsonConfiguration().jsonMapper();
        var preprocessor = TestComponents.preprocessor();
        var store = new LsaModelStore(mapper, preprocessor);
        var backend = new LsaBackend(
                new LsaService(
                        new CorpusPreparation(TestComponents.loader(), preprocessor),
                        preprocessor,
                        new LsaTrainer(),
                        store,
                        new RuntimeMetadata(mapper)
                )
        );
        var output = directory.resolve("model with spaces");
        var summary = backend.train(
                Path.of("examples/related-documents.csv"),
                output,
                6,
                new LsaParameters(3, 20000, 1)
        );
        assertThat(summary.documents()).isEqualTo(6);
        assertThat(summary.dimensions()).isEqualTo(3);
        assertThat(summary.zeroDocuments()).isZero();
        var first = backend.open(output).search("Ракеты спутники абракадабравыв", 6);
        assertThat(first.status()).isEqualTo("ok");
        assertThat(first.knownTokens()).isEqualTo(2);
        assertThat(first.totalTokens()).isEqualTo(3);
        assertThat(first.hits().subList(0, 2)).extracting(hit -> hit.id())
                .containsExactly("space-a", "space-b");
        assertThat(first.hits().getFirst().cosine()).isCloseTo(1, within(1e-9));
        assertThat(backend.open(output).search("Ракеты спутники абракадабравыв", 6)).isEqualTo(first);
        assertThat(backend.open(output).search("абракадабравыв", 3).status()).isEqualTo("oov");
        assertThat(backend.open(output).search("и в на !", 3).status()).isEqualTo("empty_after_normalization");
        assertThatThrownBy(() -> backend.open(output).search("космос", 0)).hasMessageContaining("top");
        assertThatThrownBy(
                () -> backend.train(
                        Path.of("examples/related-documents.csv"),
                        output,
                        6,
                        new LsaParameters(3, 20000, 1)
                )
        ).hasMessageContaining("уже существует");

        var dataPath = output.resolve("lsa.json.gz");
        LsaModelData data;

        try (var stream = new GZIPInputStream(Files.newInputStream(dataPath))) {
            data = mapper.readValue(stream, LsaModelData.class);
        }

        assertThat(data.vocabulary()).doesNotContain("абракадабравыв");
        var beforeSave = SearchSession.create(
                data.corpus().documents(),
                preprocessor,
                new LsaVectorizer(data.vocabulary(), data.idf(), data.projection())
        )
                .search("Ракеты спутники абракадабравыв", 6);
        assertThat(beforeSave).isEqualTo(first);

        var session = backend.open(output);

        try (var executor = Executors.newFixedThreadPool(3)) {
            var searches = java.util.stream.IntStream.range(0, 12)
                    .mapToObj(i -> executor.submit(() -> session.search("Ракеты спутники абракадабравыв", 6))).toList();

            for (var search : searches) {
                assertThat(search.get(10, TimeUnit.SECONDS)).isEqualTo(first);
            }
        }

        var manifestPath = output.resolve("manifest.json");
        var savedManifest = mapper.readValue(manifestPath.toFile(), LsaManifest.class);
        assertThat(savedManifest.algorithm()).isEqualTo("lsa");
        assertThat(savedManifest.documents()).isEqualTo(summary.documents());
        assertThat(savedManifest.vocabularySize()).isEqualTo(summary.vocabularySize());
        assertThat(savedManifest.dimensions()).isEqualTo(summary.dimensions());
        assertThat(savedManifest.dataSha256()).isEqualTo(FileChecksums.hash(dataPath));
        var manifest = (ObjectNode) mapper.readTree(manifestPath.toFile());
        assertThat(manifest.has("corpus")).isFalse();
        assertThat(manifest.has("projection")).isFalse();
        manifest.put("algorithm", "word2vec");
        mapper.writeValue(manifestPath.toFile(), manifest);
        assertThatThrownBy(() -> store.load(output)).hasMessageContaining("Ожидался комплект LSA");
        manifest.put("algorithm", "lsa");
        manifest.put("dimensions", 99);
        mapper.writeValue(manifestPath.toFile(), manifest);
        assertThatThrownBy(() -> store.load(output)).hasMessageContaining("Манифест не соответствует");
        manifest.put("dimensions", 3);
        mapper.writeValue(manifestPath.toFile(), manifest);

        var incompatible = (ObjectNode) mapper.valueToTree(data);
        incompatible.put("normalization", "different-profile");

        try (var stream = new GZIPOutputStream(Files.newOutputStream(dataPath))) {
            mapper.writeValue(stream, incompatible);
        }

        manifest.put("dataSha256", FileChecksums.hash(dataPath));
        mapper.writeValue(manifestPath.toFile(), manifest);
        assertThatThrownBy(() -> store.load(output)).hasMessageContaining("профиль нормализации LSA");

        data.projection()[0][0] = Double.NaN;

        try (var stream = new GZIPOutputStream(Files.newOutputStream(dataPath))) {
            mapper.writeValue(stream, data);
        }

        manifest.put("dataSha256", FileChecksums.hash(dataPath));
        mapper.writeValue(manifestPath.toFile(), manifest);
        assertThatThrownBy(() -> store.load(output)).hasMessageContaining("Некорректная SVD-проекция");

        Files.writeString(dataPath, "damaged", StandardOpenOption.APPEND);
        assertThatThrownBy(() -> store.load(output)).hasMessageContaining("Повреждены данные LSA");
    }

    @ParameterizedTest
    @ValueSource(strings = {"projection", "idf"})
    void rejectsIncompleteOrInconsistentPayloadEvenWithCorrectChecksum(String field) throws Exception {
        var mapper = new JsonConfiguration().jsonMapper();
        var preprocessor = TestComponents.preprocessor();
        var store = new LsaModelStore(mapper, preprocessor);
        var corpus = new Corpus(List.of(new Document("a", "", "космос"), new Document("b", "", "спорт")), 2, 0, 0);
        var model = new LsaModel(
                List.of("космос", "спорт"),
                new double[]{1, 1},
                new double[][]{{1}, {0}},
                new double[]{1},
                new LsaParameters(1, 2, 1)
        );
        var bundle = new ModelBundle<>(
                corpus,
                model,
                preprocessor.profile(),
                preprocessor.resourceHash(),
                Map.of(),
                2L,
                1L,
                Map.of()
        );
        var output = directory.resolve("model");
        store.save(output, bundle);
        var path = output.resolve("lsa.json.gz");
        ObjectNode data;

        try (var stream = new GZIPInputStream(Files.newInputStream(path))) {
            data = (ObjectNode) mapper.readTree(stream);
        }

        String expected;

        if (field.equals("projection")) {
            data.putNull("projection");
            expected = "Неполные данные LSA";
        } else {
            data.putArray("idf").add(1);
            expected = "Несогласованные размеры";
        }

        try (var stream = new GZIPOutputStream(Files.newOutputStream(path))) {
            mapper.writeValue(stream, data);
        }

        var manifestPath = output.resolve("manifest.json");
        var manifest = (ObjectNode) mapper.readTree(manifestPath.toFile());
        manifest.put("dataSha256", FileChecksums.hash(path));
        mapper.writeValue(manifestPath.toFile(), manifest);
        assertThatThrownBy(() -> store.load(output)).isInstanceOf(ModelImportException.class)
                .hasMessageContaining(expected);
    }

    @Test
    void knownTermsOutsideRetainedSubspaceReturnZeroVector() throws Exception {
        var preprocessor = TestComponents.preprocessor();
        var corpus = new Corpus(
                List.of(
                        new Document("a", "", "космос"),
                        new Document("b", "", "спорт")
                ),
                2,
                0,
                0
        );
        var data = new LsaModelData(
                corpus,
                List.of("космос", "спорт"),
                new double[]{1, 1},
                new double[][]{{1}, {0}},
                new double[]{1},
                new LsaParameters(1, 2, 1),
                preprocessor.profile(),
                preprocessor.resourceHash(),
                Map.of(),
                Map.of(),
                null,
                null
        );
        var index = SearchSession.create(
                data.corpus().documents(),
                preprocessor,
                new LsaVectorizer(data.vocabulary(), data.idf(), data.projection())
        );
        assertThat(index.zeroDocuments()).isEqualTo(1);
        assertThat(index.search("спорт", 2).status()).isEqualTo("zero_vector");
        assertThat(index.search("спорт", 2).knownTokens()).isEqualTo(1);
        assertThat(index.search("космос", 2).hits()).hasSize(1);
    }
}
