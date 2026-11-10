package ru.nlp.semantic.cnn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import ru.nlp.semantic.model.cnn.CnnParameters;
import ru.nlp.semantic.runtime.NativeRuntime;

class CnnGradientIntegrationTest {

    @Test
    void allNetworkParametersMatchFiniteDifferencesWithVariableLengthMaskedInputs() throws Exception {
        NativeRuntime.withThreads(1, () -> {
            var parameters = CnnParameters.builder()
                    .dimensions(3)
                    .embeddingDimensions(3)
                    .filters(3)
                    .kernelSize(3)
                    .maxLength(8)
                    .maxFeatures(20)
                    .minCount(1)
                    .batchSize(2)
                    .epochs(1)
                    .learningRate(0.003)
                    .temperature(0.2)
                    .seed(42)
                    .workers(1)
                    .validationFraction(0)
                    .pairSampling(PairSampling.symmetric)
                    .build();
            var network = CnnNetwork.create(parameters, 10).convertDataType(DataType.DOUBLE);
            var inputs = CnnVocabulary.inputs(
                    List.of(
                            new int[]{2, 3, 4},
                            new int[]{3, 4, 5, 6},
                            new int[]{7, 8},
                            new int[]{8, 9, 10, 11, 7}
                    )
            );

            try (
                    var originalFeatures = inputs.features();
                    var originalMask = inputs.mask();
                    var features = originalFeatures.castTo(DataType.DOUBLE);
                    var labels = Nd4j.createFromArray(new double[][]{{1}, {0}, {3}, {2}});
                    var mask = originalMask.castTo(DataType.DOUBLE)
            ) {
                var data = new DataSet(features, labels, mask, null);
                network.setInput(data.getFeatures());
                network.setLabels(data.getLabels());
                network.setLayerMaskArrays(data.getFeaturesMaskArray(), null);
                network.computeGradientAndScore();
                var gradients = network.gradient().gradientForVariable();
                assertThat(gradients.keySet()).containsExactlyInAnyOrder("0_W", "1_W", "1_b", "3_W", "3_b");

                for (var entry : network.paramTable().entrySet()) {
                    // Веса принадлежат сети; снимок градиента нужен до следующих forward-проходов.
                    var weights = entry.getValue();

                    try (var analytical = gradients.get(entry.getKey()).dup()) {
                        for (long index = 0; index < weights.length(); index++) {
                            var coordinates = coordinates(weights.shape(), index);
                            double original = weights.getDouble(coordinates);
                            double epsilon = 1e-5;
                            double plus;
                            double minus;

                            try {
                                weights.putScalar(coordinates, original + epsilon);
                                plus = network.score(data, false);
                                weights.putScalar(coordinates, original - epsilon);
                                minus = network.score(data, false);
                            } finally {
                                weights.putScalar(coordinates, original);
                            }

                            // Score — среднее; до updater DL4J хранит градиент суммы по пакету.
                            double numerical = (plus - minus) * data.numExamples() / (2 * epsilon);
                            assertThat(analytical.getDouble(coordinates))
                                    .as("%s, координата %d", entry.getKey(), index)
                                    .isCloseTo(numerical, within(1e-6));
                        }
                    }
                }

                network.clearLayerMaskArrays();
            }

            return null;
        });
    }

    private static long[] coordinates(long[] shape, long index) {
        var result = new long[shape.length];

        for (int i = shape.length - 1; i >= 0; i--) {
            result[i] = index % shape[i];
            index /= shape[i];
        }

        return result;
    }
}
