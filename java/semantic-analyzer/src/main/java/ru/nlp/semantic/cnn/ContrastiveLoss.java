package ru.nlp.semantic.cnn;

import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.shade.jackson.annotation.JsonCreator;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import lombok.Getter;

/** Симметричная NT-Xent: одна положительная пара и остальные строки пакета как отрицательные. */
public final class ContrastiveLoss implements ILossFunction {

    private static final double NORM_EPSILON = 1e-12;

    /** Температура; getter входит в JSON конфигурации DL4J. */
    @Getter
    private final double temperature;

    /** Создаёт сериализуемую функцию потерь с фиксированной температурой. */
    @JsonCreator
    public ContrastiveLoss(@JsonProperty("temperature") double temperature) {
        if (!Double.isFinite(temperature) || temperature <= 0) {
            throw new IllegalArgumentException("Температура NT-Xent должна быть конечной и > 0");
        }
        this.temperature = temperature;
    }

    /** Вычисляет сумму потерь; при average делит её на число строк. */
    @Override
    public double computeScore(
            INDArray labels,
            INDArray preOutput,
            IActivation activation,
            INDArray mask,
            boolean average
    ) {
        try (var scores = computeScoreArray(labels, preOutput, activation, mask)) {
            double sum = scores.sumNumber().doubleValue();

            return average ? sum / preOutput.size(0) : sum;
        }
    }

    /** Возвращает вклад каждой строки; labels содержит индекс её положительного партнёра. */
    @Override
    public INDArray computeScoreArray(INDArray labels, INDArray preOutput, IActivation activation, INDArray mask) {
        return calculate(labels, preOutput, activation, mask, false).scores();
    }

    /** Возвращает градиент суммы потерь; усреднение пакета выполняет DL4J. */
    @Override
    public INDArray computeGradient(INDArray labels, INDArray preOutput, IActivation activation, INDArray mask) {
        var result = calculate(labels, preOutput, activation, mask, true);

        try (var scores = result.scores()) {
            return result.gradient();
        }
    }

    /** Вычисляет score и градиент за один проход без изменения входных массивов. */
    @Override
    public Pair<Double, INDArray> computeGradientAndScore(
            INDArray labels,
            INDArray preOutput,
            IActivation activation,
            INDArray mask,
            boolean average
    ) {
        var result = calculate(labels, preOutput, activation, mask, true);

        try (var scores = result.scores()) {
            double sum = scores.sumNumber().doubleValue();

            return Pair.of(average ? sum / preOutput.size(0) : sum, result.gradient());
        }
    }

    /** Возвращает имя функции для диагностики обучения. */
    @Override
    public String name() {
        return "symmetric-nt-xent";
    }

    private Calculation calculate(
            INDArray labels,
            INDArray preOutput,
            IActivation activation,
            INDArray mask,
            boolean withGradient
    ) {
        if (
            mask != null || preOutput.rank() != 2 || preOutput.size(0) < 4 || preOutput.size(0) % 2 != 0
                    || labels.rank() != 2 || labels.size(0) != preOutput.size(0) || labels.size(1) != 1
        ) {
            throw new IllegalArgumentException("NT-Xent: нужны >= 2 полные пары, labels [строки,1], без маски потерь");
        }
        double[][] values;

        try (var activated = preOutput.dup()) {
            values = activation.getActivation(activated, true).toDoubleMatrix();
        }

        int rows = values.length;
        int dimensions = values[0].length;
        int[] partners = new int[rows];
        double[] norms = new double[rows];
        double[][] normalized = new double[rows][dimensions];

        for (int i = 0; i < rows; i++) {
            double partner = labels.getDouble(i, 0);

            if (
                !Double.isFinite(partner) || partner != Math.rint(partner) || partner < 0 || partner >= rows
                        || partner == i
            ) {
                throw new IllegalArgumentException("NT-Xent: некорректный положительный партнёр");
            }
            partners[i] = (int) partner;
            double squared = NORM_EPSILON;

            for (double value : values[i]) {
                squared += value * value;
            }

            norms[i] = Math.sqrt(squared);

            if (!Double.isFinite(norms[i])) {
                throw new IllegalArgumentException("NT-Xent: неконечный выход сети");
            }

            for (int d = 0; d < dimensions; d++) {
                normalized[i][d] = values[i][d] / norms[i];
            }
        }

        for (int i = 0; i < rows; i++) {
            if (partners[partners[i]] != i) {
                throw new IllegalArgumentException("NT-Xent: пары должны быть взаимными");
            }
        }

        double[][] scores = new double[rows][1];
        double[][] derivative = new double[rows][dimensions];

        for (int i = 0; i < rows; i++) {
            double[] logits = new double[rows];
            double maximum = Double.NEGATIVE_INFINITY;

            for (int j = 0; j < rows; j++) {
                if (i == j) {
                    continue;
                }

                for (int d = 0; d < dimensions; d++) {
                    logits[j] += normalized[i][d] * normalized[j][d] / temperature;
                }

                maximum = Math.max(maximum, logits[j]);
            }

            double denominator = 0;

            for (int j = 0; j < rows; j++) {
                if (i != j) {
                    denominator += Math.exp(logits[j] - maximum);
                }
            }

            scores[i][0] = maximum + Math.log(denominator) - logits[partners[i]];

            if (!withGradient) {
                continue;
            }

            for (int j = 0; j < rows; j++) {
                if (i == j) {
                    continue;
                }
                double coefficient = (Math.exp(logits[j] - maximum) / denominator
                        - (j == partners[i] ? 1 : 0)) / temperature;

                for (int d = 0; d < dimensions; d++) {
                    derivative[i][d] += coefficient * normalized[j][d];
                    derivative[j][d] += coefficient * normalized[i][d];
                }
            }
        }

        INDArray gradient = null;

        if (withGradient) {
            for (int i = 0; i < rows; i++) {
                double projection = 0;

                for (int d = 0; d < dimensions; d++) {
                    projection += derivative[i][d] * normalized[i][d];
                }

                for (int d = 0; d < dimensions; d++) {
                    derivative[i][d] = (derivative[i][d] - normalized[i][d] * projection) / norms[i];
                }
            }

            try (var inputCopy = preOutput.dup(); var activationGradient = array(derivative, preOutput.dataType())) {
                var propagated = activation.backprop(inputCopy, activationGradient).getFirst();
                // Активация может вернуть view временного входа: DL4J получает независимый буфер.
                gradient = propagated.dup();
            }
        }

        return new Calculation(array(scores, preOutput.dataType()), gradient);
    }

    private static INDArray array(double[][] values, DataType dataType) {
        int columns = values[0].length;
        double[] flattened = new double[values.length * columns];

        for (int i = 0; i < values.length; i++) {
            System.arraycopy(values[i], 0, flattened, i * columns, columns);
        }

        return Nd4j.create(flattened, new long[]{values.length, columns}, dataType);
    }

    private record Calculation(
            INDArray scores,
            INDArray gradient
    ) {}
}
