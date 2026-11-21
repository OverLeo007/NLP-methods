package ru.nlp.semantic.model.cnn.training;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import ru.nlp.semantic.model.cnn.dto.CnnParameters;
import ru.nlp.semantic.model.cnn.network.CnnNetwork;
import ru.nlp.semantic.model.shared.training.DocumentPairs;
import ru.nlp.semantic.model.shared.training.PairSampling;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;
import ru.nlp.semantic.runtime.NativeRuntime;

class CnnLearningTest {

    @ParameterizedTest
    @EnumSource(PairSampling.class)
    void pairsAreReproducibleUseDistinctSourcesAndExcludeDuplicateNegativeSpans(PairSampling sampling) {
        var documents = List.of(
                new int[]{2, 3, 4, 5},
                new int[]{6, 7},
                new int[]{8},
                new int[]{2, 3, 4, 5},
                new int[]{1, 1}
        );
        var pairs = new DocumentPairs(documents, 3, 3, sampling);
        assertThat(pairs.documents()).isEqualTo(3);
        var first = pairs.batches(new Random(42)).getFirst();
        var second = pairs.batches(new Random(42)).getFirst();
        assertThat(first.sources()).isEqualTo(second.sources());
        var previous = new HashSet<List<Integer>>();
        var sources = new HashSet<Integer>();

        for (int i = 0; i < first.rows().size(); i += 2) {
            assertThat(sources.add(first.sources().get(i))).isTrue();
            assertThat(first.sources().get(i)).isEqualTo(first.sources().get(i + 1));
            assertThat(first.rows().get(i)).containsExactly(second.rows().get(i));
            assertThat(first.rows().get(i + 1)).containsExactly(second.rows().get(i + 1));
            var left = Arrays.stream(first.rows().get(i)).boxed().toList();
            var right = Arrays.stream(first.rows().get(i + 1)).boxed().toList();
            assertThat(previous).doesNotContain(left, right);
            previous.add(left);
            previous.add(right);
            assertThat(first.rows().get(i).length).isBetween(1, 3);
        }

        assertThatThrownBy(() -> new DocumentPairs(List.of(new int[]{2}, new int[]{2}), 3, 2))
                .hasMessageContaining("два разных");
    }

    @Test
    void maskedPaddingDoesNotChangeShortTextOutputAndNetworkLearns() throws Exception {
        NativeRuntime.withThreads(1, () -> {
            var parameters = parameters();
            var network = CnnNetwork.create(parameters, 12);
            int[] shortText = {2, 3};
            var alone = SequenceVocabulary.inputs(List.of(shortText));
            var batch = SequenceVocabulary.inputs(List.of(shortText, new int[]{4, 5, 6, 7, 8, 9}));
            var expected = network.output(alone.features(), false, alone.mask(), null).getRow(0).toDoubleVector();
            var actual = network.output(batch.features(), false, batch.mask(), null).getRow(0).toDoubleVector();

            for (int i = 0; i < expected.length; i++) {
                assertThat(actual[i]).isCloseTo(expected[i], within(1e-6));
            }

            return null;
        });
        var documents = List.of(
                List.of("космос", "ракета", "спутник", "орбита"),
                List.of("корабль", "планета", "космос", "ракета"),
                List.of("футбол", "команда", "матч", "стадион"),
                List.of("спорт", "гол", "футбол", "команда"),
                List.of("музыка", "концерт", "песня", "гитара"),
                List.of("оркестр", "сцена", "музыка", "концерт")
        );
        var result = new CnnTrainer().train(documents, parameters());
        assertThat(result.model().training().updates()).isPositive();
        assertThat(result.model().training().parameterChangeL2()).isGreaterThan(1e-4);
        assertThat(result.model().training().finalDiagnosticLoss())
                .isLessThan(result.model().training().initialDiagnosticLoss());
        NativeRuntime.withThreads(1, () -> {
            assertTrainedNetworkIgnoresPadding(result.model().network());

            return null;
        });
    }

    private static void assertTrainedNetworkIgnoresPadding(MultiLayerNetwork network) {
        for (int length : new int[]{1, 2, 3, 8, 11}) {
            int[] shortText = new int[length];

            for (int i = 0; i < length; i++) {
                shortText[i] = i % 12 + 2;
            }

            var alone = SequenceVocabulary.inputs(List.of(shortText));
            var batch = SequenceVocabulary
                    .inputs(List.of(shortText, new int[]{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}));

            try (
                    var aloneFeatures = alone.features();
                    var aloneMask = alone.mask();
                    var batchFeatures = batch.features();
                    var batchMask = batch.mask()
            ) {
                var expected = network.output(aloneFeatures, false, aloneMask, null).getRow(0).toDoubleVector();
                var actual = network.output(batchFeatures, false, batchMask, null).getRow(0).toDoubleVector();
                var training = network.output(batchFeatures, true, batchMask, null).getRow(0).toDoubleVector();

                for (int i = length; i < 12; i++) {
                    batchFeatures.putScalar(0, i, 13);
                }

                var changedPadding = network.output(batchFeatures, false, batchMask, null).getRow(0).toDoubleVector();

                for (int d = 0; d < expected.length; d++) {
                    assertThat(actual[d]).isCloseTo(expected[d], within(1e-6));
                    assertThat(training[d]).isCloseTo(actual[d], within(1e-6));
                    assertThat(changedPadding[d]).isCloseTo(actual[d], within(1e-6));
                }

                network.clearLayerMaskArrays();
            }
        }

        network.clearLayerMaskArrays();
    }

    private static CnnParameters parameters() {
        return CnnParameters.builder()
                .dimensions(8)
                .embeddingDimensions(8)
                .filters(8)
                .kernelSize(3)
                .maxLength(12)
                .maxFeatures(100)
                .minCount(1)
                .batchSize(6)
                .epochs(15)
                .learningRate(0.01)
                .temperature(0.2)
                .seed(42)
                .workers(1)
                .validationFraction(0)
                .pairSampling(PairSampling.symmetric)
                .build();
    }
}
