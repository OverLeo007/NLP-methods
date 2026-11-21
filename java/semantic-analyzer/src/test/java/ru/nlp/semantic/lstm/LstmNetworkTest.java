package ru.nlp.semantic.lstm;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ru.nlp.semantic.TestComponents;
import ru.nlp.semantic.config.JsonConfiguration;
import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.model.lstm.LstmParameters;
import ru.nlp.semantic.model.storage.ModelBundle;
import ru.nlp.semantic.persistence.LstmModelStore;
import ru.nlp.semantic.runtime.NativeRuntime;
import ru.nlp.semantic.search.SearchSession;
import ru.nlp.semantic.training.LstmTrainer;
import ru.nlp.semantic.vectorization.LstmVectorizer;

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
    void restoresIdenticalRankingsFromSavedWeights(@TempDir Path directory) throws Exception {
        var preprocessor = TestComponents.preprocessor();
        var prepared = new CorpusPreparation(TestComponents.loader(), preprocessor)
                .prepare(Path.of("examples/related-documents.csv"), 6);
        var model = new LstmTrainer().train(prepared.documents(), PARAMETERS).model();
        assertThat(model.finalTokenLoss()).isLessThan(model.initialTokenLoss());
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
        var loaded = store.load(path);
        var reopened = SearchSession
                .create(loaded.corpus().documents(), preprocessor, new LstmVectorizer(loaded.model()));
        assertThat(reopened.search("Ракеты", 3)).isEqualTo(expected);
        assertThat(reopened.search("абракадабравыв", 3).status()).isEqualTo("oov");
        assertThat(reopened.search("", 3).status()).isEqualTo("empty_after_normalization");
    }
}
