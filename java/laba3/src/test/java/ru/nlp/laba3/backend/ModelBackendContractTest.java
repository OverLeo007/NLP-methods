package ru.nlp.laba3.backend;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.slf4j.LoggerFactory;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.corpus.CorpusPreparation;
import ru.nlp.laba3.exception.ModelImportException;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.model.word2vec.Word2VecParameters;
import ru.nlp.laba3.persistence.LsaModelStore;
import ru.nlp.laba3.persistence.ModelStore;
import ru.nlp.laba3.persistence.Word2VecModelStore;
import ru.nlp.laba3.runtime.RuntimeMetadata;
import ru.nlp.laba3.service.LsaService;
import ru.nlp.laba3.training.LsaTrainer;

import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.read.ListAppender;

class ModelBackendContractTest {

    @TempDir
    Path directory;

    @Test
    void word2VecSupportsTypedTrainingStorageAndRepeatedSearch() throws Exception {
        var store = new Word2VecModelStore(new JsonConfiguration().jsonMapper(), TestComponents.preprocessor());
        verifyLoggedRoundtrip(
                Algorithm.word2vec,
                TestComponents.word2VecBackend(),
                new Word2VecParameters(8, 2, 1, 3, 42, 1),
                store
        );
    }

    @Test
    void lsaSupportsSameContractWithoutExternalNativeLock() throws Exception {
        var preprocessor = TestComponents.preprocessor();
        var mapper = new JsonConfiguration().jsonMapper();
        var store = new LsaModelStore(mapper, preprocessor);
        ModelBackend<LsaParameters> backend = new LsaBackend(
                new LsaService(
                        new CorpusPreparation(TestComponents.loader(), preprocessor),
                        preprocessor,
                        new LsaTrainer(),
                        store,
                        new RuntimeMetadata(mapper)
                )
        );
        verifyLoggedRoundtrip(Algorithm.lsa, backend, new LsaParameters(3, 20000, 1), store);
    }

    private <P, M> void verifyLoggedRoundtrip(
            Algorithm algorithm,
            ModelBackend<P> backend,
            P parameters,
            ModelStore<M> store
    ) throws Exception {
        var logger = (Logger) LoggerFactory.getLogger("ru.nlp.laba3");
        var appender = new ListAppender<ILoggingEvent>();
        appender.start();
        logger.addAppender(appender);

        try {
            verifyRoundtrip(backend, parameters, store);
            var missing = directory.resolve("missing model");
            assertThatThrownBy(() -> store.load(missing)).isInstanceOf(ModelImportException.class)
                    .hasMessageContaining(algorithm.displayName()).hasMessageContaining(missing.toString())
                    .hasCauseInstanceOf(IOException.class);
        } finally {
            logger.detachAppender(appender);
            appender.stop();
        }

        var messages = appender.list.stream()
                .filter(
                        event -> event.getLoggerName().startsWith("ru.nlp.laba3.service.")
                                || event.getLoggerName().startsWith("ru.nlp.laba3.training.")
                                || event.getLoggerName().startsWith("ru.nlp.laba3.persistence.")
                )
                .map(ILoggingEvent::getFormattedMessage).toList();
        assertThat(messages).isNotEmpty()
                .allSatisfy(message -> assertThat(message).startsWith("[" + algorithm.displayName() + "] "));
        assertThat(messages).anySatisfy(message -> assertThat(message).contains("Обучение модели завершено"));
        assertThat(messages).anySatisfy(message -> assertThat(message).contains("Сохранение модели"));
        assertThat(messages).anySatisfy(message -> assertThat(message).contains("Модель загружена"));
    }

    private <P, M> void verifyRoundtrip(ModelBackend<P> backend, P parameters, ModelStore<M> store) throws Exception {
        var input = Path.of("examples/related-documents.csv");
        var output = directory.resolve("model with spaces");
        var summary = backend.train(input, output, 6, parameters);
        assertThat(summary.documents()).isEqualTo(6);
        assertThat(summary.dimensions()).isPositive();
        assertThat(summary.vocabularySize()).isPositive();
        var loaded = store.load(output);
        assertThat(loaded.corpus()).isEqualTo(TestComponents.loader().read(input, 6));
        assertThat(loaded.tokens()).isEqualTo(summary.tokens());
        assertThat(loaded.zeroDocuments()).isEqualTo(summary.zeroDocuments());
        assertThat(loaded.normalization()).isEqualTo(TestComponents.preprocessor().profile());
        assertThat(loaded.versions()).containsKeys("java", "nd4jBackend");
        assertThat(loaded.seconds()).containsOnlyKeys("preparation", "training", "indexing", "saving");
        assertThat(loaded.seconds()).isEqualTo(summary.seconds());
        var opened = backend.openModel(output);
        assertThat(opened.metadata().parameters()).isEqualTo(parameters);
        assertThat(opened.metadata().versions()).isEqualTo(loaded.versions());
        assertThat(opened.metadata().seconds()).isEqualTo(loaded.seconds());
        assertThat(opened.session().documents()).isEqualTo(loaded.corpus().documents());
        assertThat(opened.session().zeroDocumentIds()).hasSize((int) opened.session().zeroDocuments());
        assertThatThrownBy(() -> opened.session().documents().clear())
                .isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> opened.session().zeroDocumentIds().clear())
                .isInstanceOf(UnsupportedOperationException.class);
        assertThat(summary.trainingSeconds()).isEqualTo(summary.seconds().get("training"));
        assertThat(summary.seconds().values())
                .allSatisfy(value -> assertThat(value).isFinite().isGreaterThanOrEqualTo(0));
        assertThatThrownBy(() -> summary.seconds().put("training", -1.0))
                .isInstanceOf(UnsupportedOperationException.class);

        String query = loaded.corpus().documents().getFirst().text();
        var expected = backend.open(output).search(query, 6);
        assertThat(expected.status()).isEqualTo("ok");
        assertThat(expected.hits().getFirst().id()).isEqualTo(loaded.corpus().documents().getFirst().id());
        assertThat(expected.hits().getFirst().cosine()).isCloseTo(1, within(1e-6));

        var copy = directory.resolve("copied model");
        var savedTimings = store.save(copy, loaded);
        var copied = store.load(copy);
        assertThat(copied.tokens()).isEqualTo(loaded.tokens());
        assertThat(copied.zeroDocuments()).isEqualTo(loaded.zeroDocuments());
        assertThat(copied.seconds()).containsAllEntriesOf(
                loaded.seconds().entrySet().stream()
                        .filter(entry -> !entry.getKey().equals("saving"))
                        .collect(
                                java.util.stream.Collectors
                                        .toMap(java.util.Map.Entry::getKey, java.util.Map.Entry::getValue)
                        )
        );
        assertThat(copied.seconds()).isEqualTo(savedTimings);
        assertThat(copied.versions()).isEqualTo(loaded.versions());
        var session = backend.open(copy);
        var actual = session.search(query, 6);
        assertThat(actual).isEqualTo(expected);
        assertThat(session.search("абракадабравыв", 6).status()).isEqualTo("oov");
        assertThat(session.search("и в на !", 6).status()).isEqualTo("empty_after_normalization");

        try (var executor = Executors.newFixedThreadPool(3)) {
            var searches = java.util.stream.IntStream.range(0, 12)
                    .mapToObj(ignored -> executor.submit(() -> session.search(query, 6))).toList();

            for (var search : searches) {
                assertThat(search.get(10, TimeUnit.SECONDS)).isEqualTo(actual);
            }
        }

        byte[] manifest = Files.readAllBytes(output.resolve("manifest.json"));
        assertThatThrownBy(() -> backend.train(input, output, 6, parameters)).hasMessageContaining(output.toString());
        assertThatThrownBy(() -> store.save(output, loaded)).hasMessageContaining(output.toString());
        assertThat(Files.readAllBytes(output.resolve("manifest.json"))).isEqualTo(manifest);
    }
}
