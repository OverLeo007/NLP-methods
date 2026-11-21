package ru.nlp.semantic.model.lstm.network;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import com.fasterxml.jackson.databind.node.ObjectNode;

import ru.nlp.semantic.TestComponents;
import ru.nlp.semantic.config.JsonConfiguration;
import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.model.base.dto.ModelBundle;
import ru.nlp.semantic.model.lstm.dto.LstmObjective;
import ru.nlp.semantic.model.lstm.dto.LstmParameters;
import ru.nlp.semantic.model.lstm.store.LstmModelStore;
import ru.nlp.semantic.model.lstm.training.LstmTrainer;
import ru.nlp.semantic.model.lstm.vectorization.LstmVectorizer;
import ru.nlp.semantic.model.shared.training.DocumentPairs;
import ru.nlp.semantic.model.word2vec.training.Word2VecTrainer;
import ru.nlp.semantic.runtime.NativeRuntime;
import ru.nlp.semantic.search.SearchSession;

class LstmNetworkTest {

    private static final LstmParameters PARAMETERS = new LstmParameters(8, 8, 12, 100, 1, 6, 20, 0.01, 42, 1);

    @Test
    void masksPaddingAndUnknownTargetsWhileTrainingBothRecurrentLayers() throws Exception {
        NativeRuntime.withThreads(1, () -> {
            var graph = LstmNetwork.create(PARAMETERS, 3);
            var batch = LstmNetwork.batch(List.of(new int[]{2, 1}, new int[]{4, 3, 2}), 3);
            assertThat(batch.getFeaturesMaskArray().getRow(0).toDoubleVector()).containsExactly(1, 1, 0);
            assertThat(batch.getLabelsMaskArray().getRow(0).toDoubleVector()).containsExactly(1, 0, 0);
            var context = LstmNetwork.encode(graph, batch.getFeatures(), batch.getFeaturesMaskArray()).dup();
            var score = graph.score(batch);
            batch.getFeatures().putScalar(0, 0, 2, 3);
            batch.getLabels().putScalar(0, 1, 1, 1);
            batch.getLabels().putScalar(0, 2, 2, 1);
            assertThat(graph.score(batch)).isCloseTo(score, within(1e-6));
            assertThat(
                    LstmNetwork.encode(graph, batch.getFeatures(), batch.getFeaturesMaskArray())
                            .equalsWithEps(context, 1e-6)
            ).isTrue();

            var encoder = graph.getLayer("encoder").getParam("W").dup();
            var decoder = graph.getLayer("decoder").getParam("W").dup();
            graph.fit(batch);
            assertThat(graph.getLayer("encoder").getParam("W").distance2(encoder)).isPositive();
            assertThat(graph.getLayer("decoder").getParam("W").distance2(decoder)).isPositive();

            return null;
        });
    }

    @Test
    void retrievalPoolingIgnoresMaskedStepsAndPairLossTrainsEncoder() throws Exception {
        NativeRuntime.withThreads(1, () -> {
            var parameters = new LstmParameters(8, 8, 12, 100, 1, 2, 1, 0.01, 42, 1, LstmObjective.retrieval);
            var graph = LstmNetwork.create(parameters, 3);
            var batch = new DocumentPairs.Batch(
                    List.of(new int[]{2}, new int[]{2, 3}, new int[]{4, 4}, new int[]{4}),
                    List.of(0, 0, 1, 1)
            ).dataSet();
            batch.setFeatures(batch.getFeatures().reshape(4, 1, 2));
            var mask = batch.getFeaturesMaskArray();
            var context = LstmNetwork.encode(graph, batch.getFeatures(), mask);
            var single = LstmNetwork.batch(List.of(new int[]{2}), 3);
            var unpadded = LstmNetwork.encode(graph, single.getFeatures(), single.getFeaturesMaskArray());
            assertThat(context.getRow(0).equalsWithEps(unpadded.getRow(0), 1e-6)).isTrue();
            var before = graph.getLayer("encoder").getParam("RW").dup();
            graph.fit(batch);
            assertThat(graph.getLayer("encoder").getParam("RW").distance2(before)).isPositive();

            return null;
        });
    }

    @ParameterizedTest
    @EnumSource(LstmObjective.class)
    void restoresIdenticalRankingsFromSavedWeights(LstmObjective objective, @TempDir Path directory) throws Exception {
        var preprocessor = TestComponents.preprocessor();
        var prepared = new CorpusPreparation(TestComponents.loader(), preprocessor)
                .prepare(Path.of("examples/related-documents.csv"), 6);
        var parameters = new LstmParameters(8, 8, 12, 100, 1, 6, 20, 0.01, 42, 1, objective);
        var model = new LstmTrainer(new Word2VecTrainer()).train(prepared.documents(), parameters).model();
        assertThat(model.finalLoss()).isLessThan(model.initialLoss());
        var session = SearchSession.create(prepared.corpus().documents(), preprocessor, new LstmVectorizer(model));
        var expected = session.search("Ракеты", 3);
        assertThat(expected.status()).isEqualTo("ok");
        var store = new LstmModelStore(new JsonConfiguration().jsonMapper(), preprocessor);
        var path = directory.resolve("модель");
        store.save(
                path,
                new ModelBundle<>(
                        prepared.corpus(),
                        model,
                        preprocessor.profile(),
                        preprocessor.resourceHash(),
                        Map.of(),
                        prepared.tokens(),
                        session.zeroDocuments(),
                        Map.of()
                )
        );
        var mapper = new JsonConfiguration().jsonMapper();
        var manifestFile = path.resolve("manifest.json");
        var manifest = mapper.readTree(Files.readAllBytes(manifestFile));

        if (objective == LstmObjective.seq2seq) {
            assertThat(manifest.has("initialTokenLoss")).isTrue();
            assertThat(manifest.has("initialPairLoss")).isFalse();
            ((ObjectNode) manifest.path("parameters")).remove("objective");
            Files.write(manifestFile, mapper.writeValueAsBytes(manifest));
        } else {
            assertThat(manifest.has("initialTokenLoss")).isFalse();
            assertThat(manifest.has("initialPairLoss")).isTrue();
        }
        var loaded = store.load(path);
        assertThat(loaded.model().parameters().objective()).isEqualTo(objective);
        var reopened = SearchSession
                .create(loaded.corpus().documents(), preprocessor, new LstmVectorizer(loaded.model()));
        assertThat(reopened.search("Ракеты", 3)).isEqualTo(expected);
        assertThat(reopened.search("абракадабравыв", 3).status()).isEqualTo("oov");
        assertThat(reopened.search("", 3).status()).isEqualTo("empty_after_normalization");
    }
}
