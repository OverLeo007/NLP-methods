package ru.nlp.semantic.model.shared.training;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;

class ContrastiveLossTest {

    @Test
    void lossMatchesIndependentClosedFormAndPrefersCorrectPartners() {
        var loss = new ContrastiveLoss(0.2);
        var activation = Activation.IDENTITY.getActivationFunction();

        try (
                var values = Nd4j.createFromArray(new double[][]{{1, 0}, {1, 0}, {0, 1}, {0, 1}});
                var labels = Nd4j.createFromArray(new double[][]{{1}, {0}, {3}, {2}});
                var wrongPartners = Nd4j.createFromArray(new double[][]{{2}, {3}, {0}, {1}});
                var identical = Nd4j.ones(DataType.DOUBLE, 4, 2)
        ) {
            double expected = Math.log1p(2 * Math.exp(-5));
            assertThat(loss.computeScore(labels, values, activation, null, true))
                    .isCloseTo(expected, within(1e-10));
            assertThat(loss.computeScore(wrongPartners, values, activation, null, true))
                    .isCloseTo(expected + 5, within(1e-10));
            assertThat(loss.computeScore(labels, identical, activation, null, true))
                    .isCloseTo(Math.log(3), within(1e-12));
        }
    }

    @ParameterizedTest
    @EnumSource(value = Activation.class, names = {"IDENTITY", "TANH"})
    void analyticalGradientMatchesFiniteDifferencesIncludingCandidateContributions(Activation activation) {
        var values = Nd4j.createFromArray(
                new double[][]{{0.4, -0.2, 0.8}, {0.3, 0.1, 0.7},
                        {-0.6, 0.2, 0.1}, {-0.4, 0.3, -0.2}}
        );
        var labels = Nd4j.createFromArray(new double[][]{{1}, {0}, {3}, {2}});
        var original = values.dup();
        var loss = new ContrastiveLoss(0.2);
        var function = activation.getActivationFunction();
        var gradient = loss.computeGradient(labels, values, function, null);
        assertThat(values).isEqualTo(original);

        for (int i = 0; i < values.rows(); i++) {
            for (int j = 0; j < values.columns(); j++) {
                double base = original.getDouble(i, j);
                values.putScalar(i, j, base + 1e-6);
                double plus = loss.computeScore(labels, values, function, null, false);
                values.putScalar(i, j, base - 1e-6);
                double minus = loss.computeScore(labels, values, function, null, false);
                values.putScalar(i, j, base);
                assertThat(gradient.getDouble(i, j)).isCloseTo((plus - minus) / 2e-6, within(1e-5));
            }
        }

        double sum = loss.computeScore(labels, values, function, null, false);
        assertThat(loss.computeScore(labels, values, function, null, true)).isCloseTo(sum / 4, within(1e-12));
        assertThat(loss.computeScoreArray(labels, values, function, null).sumNumber().doubleValue())
                .isCloseTo(sum, within(1e-12));
    }

    @Test
    void stableForLargeLogitsAndRejectsInvalidPartnersOrLossMask() {
        var loss = new ContrastiveLoss(0.01);
        var values = Nd4j.createFromArray(new double[][]{{1e10, 0}, {1e10, 0}, {0, 1e10}, {0, 1e10}});
        var labels = Nd4j.createFromArray(new double[][]{{1}, {0}, {3}, {2}});
        var activation = Activation.IDENTITY.getActivationFunction();
        assertThat(loss.computeScore(labels, values, activation, null, true)).isFinite().isGreaterThanOrEqualTo(0);
        assertThat(loss.computeGradient(labels, values, activation, null).norm2Number().doubleValue()).isFinite();
        labels.putScalar(0, 0, 0);
        assertThatThrownBy(() -> loss.computeScore(labels, values, activation, null, true))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> loss.computeScore(labels, values, activation, Nd4j.ones(DataType.DOUBLE, 4, 1), true))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
