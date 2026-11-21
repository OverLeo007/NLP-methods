package ru.nlp.semantic.lstm;

import java.util.List;

import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.rnn.DuplicateToTimeSeriesVertex;
import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import ru.nlp.semantic.model.lstm.LstmParameters;
import ru.nlp.semantic.vectorization.SequenceVocabulary;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** LSTM-автокодировщик с фиксированным вектором между кодировщиком и декодером. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class LstmNetwork {

    /** Создаёт граф для восстановления токенов из последнего действительного состояния кодировщика. */
    public static ComputationGraph create(LstmParameters parameters, int words) {
        var config = new NeuralNetConfiguration.Builder()
                .seed(parameters.seed())
                .dataType(DataType.FLOAT)
                .updater(new Adam(parameters.learningRate()))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(5)
                // DL4J берёт nIn таблицы embedding из InputType. На вход подаются ID формы [пакет, 1, длина].
                .graphBuilder()
                .addInputs("tokens")
                .setInputTypes(InputType.recurrent(words + 2))
                .addLayer(
                        "embedding",
                        new EmbeddingSequenceLayer.Builder()
                                .nIn(words + 2)
                                .nOut(parameters.embeddingDimensions())
                                .inferInputLength(true)
                                .build(),
                        "tokens"
                )
                .addLayer(
                        "encoder",
                        new LSTM.Builder()
                                .nIn(parameters.embeddingDimensions())
                                .nOut(parameters.dimensions())
                                .activation(Activation.TANH)
                                .build(),
                        "embedding"
                )
                .addVertex("context", new LastTimeStepVertex("tokens"), "encoder")
                .addVertex("repeat", new DuplicateToTimeSeriesVertex("tokens"), "context")
                .addLayer(
                        "decoder",
                        new LSTM.Builder()
                                .nIn(parameters.dimensions())
                                .nOut(parameters.dimensions())
                                .activation(Activation.TANH)
                                .build(),
                        "repeat"
                )
                .addLayer(
                        "words",
                        new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                .activation(Activation.SOFTMAX)
                                .nIn(parameters.dimensions())
                                .nOut(words)
                                .build(),
                        "decoder"
                )
                .setOutputs("words")
                .build();
        var graph = new ComputationGraph(config);
        graph.init();

        return graph;
    }

    /** Строит цели по обычным токенам. PAD и OOV не входят в функцию потерь. */
    public static DataSet batch(List<int[]> rows, int words) {
        var inputs = SequenceVocabulary.inputs(rows);
        long steps = inputs.features().size(1);
        var features = inputs.features().reshape(rows.size(), 1, steps);
        var labels = Nd4j.zeros(DataType.FLOAT, rows.size(), words, steps);
        var labelMask = Nd4j.zeros(DataType.FLOAT, rows.size(), steps);

        for (int i = 0; i < rows.size(); i++) {
            int[] ids = rows.get(i);

            for (int j = 0; j < ids.length; j++) {
                if (ids[j] >= 2) {
                    labels.putScalar(i, ids[j] - 2, j, 1);
                    labelMask.putScalar(i, j, 1);
                }
            }
        }

        return new DataSet(features, labels, inputs.mask(), labelMask);
    }

    /** Выполняет только кодировщик и возвращает состояние на последнем токене. */
    public static INDArray encode(ComputationGraph graph, INDArray features, INDArray mask) {
        return graph.output(List.of("context"), false, new INDArray[]{features}, new INDArray[]{mask})[0];
    }
}
