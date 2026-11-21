package ru.nlp.semantic.model.cnn.network;

import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Convolution1DLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.EmbeddingSequenceLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.learning.config.Adam;

import ru.nlp.semantic.exception.ModelImportException;
import ru.nlp.semantic.model.cnn.dto.CnnParameters;
import ru.nlp.semantic.model.shared.training.ContrastiveLoss;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Конфигурация одного CNN-кодировщика; рекуррентных слоёв и декодера нет. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class CnnNetwork {

    /** Создаёт сеть со встроенными обучаемыми embeddings, свёрткой и max pooling. */
    public static MultiLayerNetwork create(CnnParameters parameters, int vocabularySize) {
        var configuration = new NeuralNetConfiguration.Builder()
                .seed(parameters.seed())
                .dataType(DataType.FLOAT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(parameters.learningRate()))
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(5)
                .trainingWorkspaceMode(WorkspaceMode.NONE)
                .inferenceWorkspaceMode(WorkspaceMode.NONE)
                .list()
                .layer(
                        new EmbeddingSequenceLayer.Builder()
                                .nIn(vocabularySize + 2)
                                .nOut(parameters.embeddingDimensions())
                                .inputLength(parameters.maxLength())
                                .inferInputLength(true)
                                .activation(Activation.IDENTITY)
                                .build()
                )
                .layer(
                        new Convolution1DLayer.Builder(parameters.kernelSize())
                                .nOut(parameters.filters())
                                .convolutionMode(ConvolutionMode.Same)
                                .activation(Activation.TANH)
                                .build()
                )
                .layer(
                        new GlobalPoolingLayer.Builder(PoolingType.MAX)
                                .build()
                )
                .layer(
                        new DenseLayer.Builder()
                                .nOut(parameters.dimensions())
                                .activation(Activation.IDENTITY)
                                .build()
                )
                .layer(
                        new LossLayer.Builder(new ContrastiveLoss(parameters.temperature()))
                                .activation(Activation.IDENTITY)
                                .build()
                )
                // InputType задаёт форму последовательности индексов; RNN-слоёв в сети нет.
                .setInputType(InputType.recurrent(vocabularySize + 2, parameters.maxLength()))
                .build();
        var network = new MultiLayerNetwork(configuration);
        network.init();

        return network;
    }

    /** Проверяет соответствие загруженных слоёв манифесту до восстановления индекса. */
    public static void validate(MultiLayerNetwork network, CnnParameters parameters, int vocabularySize) {
        // Это заимствованный массив сети: его нельзя закрывать после проверки.
        var weights = network.params();

        if (
            network.getnLayers() != 5
                    || !(network.getLayer(0).conf().getLayer() instanceof EmbeddingSequenceLayer embedding)
                    || embedding.getNIn() != vocabularySize + 2
                    || embedding.getNOut() != parameters.embeddingDimensions()
                    || !(network.getLayer(1).conf().getLayer() instanceof Convolution1DLayer convolution)
                    || convolution.getNIn() != parameters.embeddingDimensions()
                    || convolution.getNOut() != parameters.filters()
                    || convolution.getKernelSize()[0] != parameters.kernelSize()
                    || convolution.getStride()[0] != 1 || convolution.getConvolutionMode() != ConvolutionMode.Same
                    || !(network.getLayer(2).conf().getLayer() instanceof GlobalPoolingLayer pooling)
                    || pooling.getPoolingType() != PoolingType.MAX
                    || !(network.getLayer(3).conf().getLayer() instanceof DenseLayer projection)
                    || projection.getNIn() != parameters.filters() || projection.getNOut() != parameters.dimensions()
                    || !(network.getLayer(4).conf().getLayer() instanceof LossLayer loss)
                    || !(loss.getLossFn() instanceof ContrastiveLoss contrastive)
                    || contrastive.getTemperature() != parameters.temperature()
                    || !Double.isFinite(weights.norm2Number().doubleValue())
        ) {
            throw new ModelImportException("CNN: веса или архитектура не соответствуют манифесту");
        }
    }
}
