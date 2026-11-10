package ru.nlp.semantic.persistence;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import ru.nlp.semantic.TestComponents;
import ru.nlp.semantic.backend.CnnBackend;
import ru.nlp.semantic.cnn.PairSampling;
import ru.nlp.semantic.config.JsonConfiguration;
import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.exception.ModelImportException;
import ru.nlp.semantic.model.cnn.CnnParameters;
import ru.nlp.semantic.runtime.RuntimeMetadata;
import ru.nlp.semantic.search.SearchSession;
import ru.nlp.semantic.training.CnnTrainer;
import ru.nlp.semantic.vectorization.CnnVectorizer;

class CnnModelPersistenceTest {

    @TempDir
    Path directory;

    @ParameterizedTest
    @EnumSource(PairSampling.class)
    void learnedModelSurvivesStorageSupportsRepeatedQueriesAndRejectsCorruption(PairSampling pairSampling)
            throws Exception {
        var preprocessor = TestComponents.preprocessor();
        var mapper = new JsonConfiguration().jsonMapper();
        var store = new CnnModelStore(mapper, preprocessor);
        var backend = new CnnBackend(
                new CorpusPreparation(TestComponents.loader(), preprocessor),
                preprocessor,
                new CnnTrainer(),
                store,
                new RuntimeMetadata(mapper)
        );
        var output = directory.resolve("модель с пробелами");
        var parameters = CnnParameters.builder()
                .dimensions(8)
                .embeddingDimensions(8)
                .filters(8)
                .kernelSize(3)
                .maxLength(12)
                .maxFeatures(100)
                .minCount(1)
                .batchSize(6)
                .epochs(5)
                .learningRate(0.01)
                .temperature(0.2)
                .seed(42)
                .workers(1)
                .validationFraction(0)
                .pairSampling(pairSampling)
                .build();
        var summary = backend.train(Path.of("examples/related-documents.csv"), output, 6, parameters);
        assertThat(summary.documents()).isEqualTo(6);
        assertThat(summary.dimensions()).isEqualTo(8);
        var bundle = store.load(output);
        assertThat(bundle.model().parameters()).isEqualTo(parameters);

        if (pairSampling == PairSampling.symmetric) {
            var legacyPath = output.resolve("manifest.json");
            byte[] originalBytes = Files.readAllBytes(legacyPath);
            var legacyManifest = mapper.readTree(originalBytes);
            ((com.fasterxml.jackson.databind.node.ObjectNode) legacyManifest.path("parameters")).remove("pairSampling");

            try {
                mapper.writeValue(legacyPath.toFile(), legacyManifest);
                assertThat(store.load(output).model().parameters().pairSampling()).isEqualTo(PairSampling.symmetric);
            } finally {
                Files.write(legacyPath, originalBytes);
            }
        }
        assertThat(bundle.model().training().updates()).isEqualTo(5);
        var vectorizer = new CnnVectorizer(bundle.model());
        var session = SearchSession.create(bundle.corpus().documents(), preprocessor, vectorizer);
        String query = bundle.corpus().documents().getFirst().text();
        var expected = session.search(query, 6);
        assertThat(expected.hits().getFirst().id()).isEqualTo(bundle.corpus().documents().getFirst().id());
        assertThat(expected.hits().getFirst().cosine()).isCloseTo(1, within(1e-6));
        var secondPath = directory.resolve("копия");
        store.save(secondPath, bundle);
        assertThat(backend.open(secondPath).search(query, 6)).isEqualTo(expected);
        assertThat(backend.openModel(output).metadata().parameters()).isEqualTo(parameters);
        assertThat(session.search("абракадабравыв", 3).status()).isEqualTo("oov");
        assertThat(session.search("и в на !", 3).status()).isEqualTo("empty_after_normalization");
        assertThat(vectorizer.encode(List.of()).isZero()).isTrue();
        var prefix = java.util.Collections.nCopies(12, "несуществующийтокен");
        var longTokens = new java.util.ArrayList<>(prefix);
        longTokens.add(preprocessor.tokens(query).getFirst());
        assertThat(vectorizer.encode(longTokens).knownTokens()).isZero();

        try (var executor = Executors.newFixedThreadPool(3)) {
            var searches = java.util.stream.IntStream.range(0, 9)
                    .mapToObj(ignored -> executor.submit(() -> session.search(query, 6))).toList();

            for (var search : searches) {
                assertThat(search.get(10, TimeUnit.SECONDS)).isEqualTo(expected);
            }
        }

        var weights = output.resolve("network.zip");
        byte[] original = Files.readAllBytes(weights);
        Files.write(weights, new byte[]{1, 2, 3});
        assertThatThrownBy(() -> store.load(output)).isInstanceOf(ModelImportException.class)
                .hasMessageContaining("Повреждены веса");
        Files.write(weights, original);
        var manifestPath = output.resolve("manifest.json");
        byte[] originalManifest = Files.readAllBytes(manifestPath);
        var manifest = mapper.readTree(originalManifest);
        ((com.fasterxml.jackson.databind.node.ObjectNode) manifest.path("parameters")).put("dimensions", 9);
        mapper.writeValue(manifestPath.toFile(), manifest);
        assertThatThrownBy(() -> store.load(output)).isInstanceOf(ModelImportException.class)
                .hasMessageContaining("архитектура");
        Files.write(manifestPath, originalManifest);
        assertThatThrownBy(() -> backend.train(Path.of("examples/related-documents.csv"), output, 6, parameters))
                .hasMessageContaining("уже существует");
        assertThat(Files.readAllBytes(manifestPath)).isEqualTo(originalManifest);
    }
}
