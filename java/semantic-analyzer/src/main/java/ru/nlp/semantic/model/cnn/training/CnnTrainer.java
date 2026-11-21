package ru.nlp.semantic.model.cnn.training;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.stereotype.Component;

import ru.nlp.semantic.exception.ModelTrainingException;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.training.AbstractModelTrainer;
import ru.nlp.semantic.model.cnn.CnnModel;
import ru.nlp.semantic.model.cnn.dto.CnnParameters;
import ru.nlp.semantic.model.cnn.dto.CnnTrainingStatistics;
import ru.nlp.semantic.model.cnn.network.CnnNetwork;
import ru.nlp.semantic.model.shared.training.DocumentPairs;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;

import lombok.extern.slf4j.Slf4j;

/** Контрастивное обучение CNN только по токенам корпуса, без оценочных запросов и qrels. */
@Component
@Slf4j
public final class CnnTrainer extends AbstractModelTrainer<CnnParameters, CnnModel> {

    /** Подключает CNN к общим границам обучения и нативных вычислений. */
    public CnnTrainer() {
        super(Algorithm.cnn);
    }

    /** Строит словарь, обучает все слои и сохраняет проверяемую диагностику. */
    @Override
    protected CnnModel fit(List<List<String>> documents, CnnParameters parameters) {
        var split = CnnTrainingSplit.create(documents, parameters.validationFraction(), parameters.seed());
        var vocabulary = SequenceVocabulary.fit(split.training(), parameters.minCount(), parameters.maxFeatures());
        var pairs = new DocumentPairs(
                split.training().stream().map(vocabulary::encode).toList(),
                parameters.maxLength(),
                parameters.batchSize(),
                parameters.pairSampling()
        );
        int validationDocuments = 0;
        List<DataSet> validation = List.of();

        if (!split.validation().isEmpty()) {
            var validationPairs = new DocumentPairs(
                    split.validation().stream().map(vocabulary::encode).toList(),
                    parameters.maxLength(),
                    parameters.batchSize(),
                    parameters.pairSampling()
            );
            var validationBatches = validationPairs.batches(new Random(parameters.seed() ^ 0x6A09E667L));
            validationDocuments = validationBatches.stream().mapToInt(batch -> batch.rows().size() / 2).sum();
            validation = validationBatches.stream().map(DocumentPairs.Batch::dataSet).toList();
        }
        var network = CnnNetwork.create(parameters, vocabulary.words().size());

        try (var initialWeights = network.params().dup()) {
            var diagnostic = pairs.batches(new Random(parameters.seed() ^ 0x5DEECE66DL)).getFirst().dataSet();
            double initialLoss = network.score(diagnostic, false);
            var losses = new ArrayList<Double>();
            var validationLosses = new ArrayList<Double>();
            int bestValidationEpoch = 0;
            var random = new Random(parameters.seed());
            int updates = 0;

            for (int epoch = 0; epoch < parameters.epochs(); epoch++) {
                double sum = 0;
                int rows = 0;

                for (var batch : pairs.batches(random)) {
                    var data = batch.dataSet();
                    network.fit(data);
                    double loss = network.score();
                    checkFinite(network, loss);
                    sum += loss * batch.rows().size();
                    rows += batch.rows().size();
                    updates++;
                }

                double loss = sum / rows;
                losses.add(loss);
                log.info("[CNN] Эпоха {}/{}: loss={}, обновлений={}", epoch + 1, parameters.epochs(), loss, updates);

                if (!validation.isEmpty()) {
                    double validationLoss = validationScore(network, validation);
                    validationLosses.add(validationLoss);

                    if (bestValidationEpoch == 0 || validationLoss < validationLosses.get(bestValidationEpoch - 1)) {
                        bestValidationEpoch = epoch + 1;
                    }
                    log.info(
                            "[CNN] Validation: документов={}, loss={}, лучшая эпоха={}",
                            validationDocuments,
                            validationLoss,
                            bestValidationEpoch
                    );
                }
            }

            double finalLoss = network.score(diagnostic, false);
            checkFinite(network, finalLoss);
            var weights = network.params();
            double change;

            try (var difference = weights.sub(initialWeights)) {
                change = difference.norm2Number().doubleValue();
            }

            var statistics = new CnnTrainingStatistics(
                    pairs.documents(),
                    updates,
                    losses,
                    initialLoss,
                    finalLoss,
                    change,
                    validationDocuments,
                    validationLosses,
                    bestValidationEpoch
            );
            log.info(
                    "[CNN] Фиксированный обучающий пакет: loss {} -> {}; изменение весов L2={}",
                    initialLoss,
                    finalLoss,
                    change
            );
            network.clearLayerMaskArrays();

            return new CnnModel(network, vocabulary.words(), parameters, statistics);
        }
    }

    private static double validationScore(MultiLayerNetwork network, List<DataSet> batches) {
        double sum = 0;
        long rows = 0;

        for (var batch : batches) {
            long batchRows = batch.getFeatures().size(0);
            double score = network.score(batch, false);

            if (!Double.isFinite(score)) {
                throw new ModelTrainingException("CNN: неконечная validation-потеря");
            }
            sum += score * batchRows;
            rows += batchRows;
        }

        return sum / rows;
    }

    /** Возвращает число потоков из типизированных параметров CNN. */
    @Override
    protected int workers(CnnParameters parameters) {
        return parameters.workers();
    }

    /** Возвращает число обычных терминов без PAD/OOV. */
    @Override
    protected int vocabularySize(CnnModel model) {
        return model.vocabulary().size();
    }

    /** Возвращает размер представления текста. */
    @Override
    protected int dimensions(CnnModel model) {
        return model.parameters().dimensions();
    }

    private static void checkFinite(MultiLayerNetwork network, double loss) {
        // Веса принадлежат сети и остаются открытыми для следующих обновлений и поиска.
        var weights = network.params();

        if (!Double.isFinite(loss) || !Double.isFinite(weights.norm2Number().doubleValue())) {
            throw new ModelTrainingException("CNN: обучение получило неконечные веса или потерю");
        }
    }
}
