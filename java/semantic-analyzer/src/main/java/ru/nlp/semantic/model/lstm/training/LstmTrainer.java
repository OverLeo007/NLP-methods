package ru.nlp.semantic.model.lstm.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.DataSet;
import org.springframework.stereotype.Component;

import ru.nlp.semantic.exception.InvalidInputException;
import ru.nlp.semantic.exception.ModelTrainingException;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.training.AbstractModelTrainer;
import ru.nlp.semantic.model.lstm.LstmModel;
import ru.nlp.semantic.model.lstm.dto.LstmObjective;
import ru.nlp.semantic.model.lstm.dto.LstmParameters;
import ru.nlp.semantic.model.lstm.network.LstmNetwork;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;
import ru.nlp.semantic.model.word2vec.training.Word2VecTrainer;

import lombok.extern.slf4j.Slf4j;

/** Обучение LSTM по парам фрагментов или по задаче восстановления токенов. */
@Component
@Slf4j
public final class LstmTrainer extends AbstractModelTrainer<LstmParameters, LstmModel> {

    private final Word2VecTrainer word2Vec;

    /** Подключает обучение к общему нативному контексту. */
    public LstmTrainer(Word2VecTrainer word2Vec) {
        super(Algorithm.lstm);
        this.word2Vec = word2Vec;
    }

    @Override
    protected LstmModel fit(List<List<String>> documents, LstmParameters parameters) {
        if (parameters.objective() == LstmObjective.retrieval) {
            return RetrievalLstmTraining.fit(documents, parameters, word2Vec);
        }
        var prefixes = documents.stream()
                .map(tokens -> tokens.subList(0, Math.min(tokens.size(), parameters.maxLength()))).toList();
        var vocabulary = SequenceVocabulary.fit(prefixes, parameters.minCount(), parameters.maxFeatures());
        var rows = new ArrayList<>(
                prefixes.stream().map(vocabulary::encode)
                        .filter(ids -> SequenceVocabulary.knownTokens(ids) > 0).toList()
        );

        if (rows.size() < 2) {
            throw new InvalidInputException("LSTM: нужны минимум два документа с известными токенами");
        }
        int words = vocabulary.words().size();
        int batchSize = Math.min(parameters.batchSize(), rows.size());
        int length = rows.stream().mapToInt(row -> row.length).max().orElseThrow();
        long cells = (long) batchSize * length
                * Math.max(Math.max(words, parameters.dimensions()), parameters.embeddingDimensions());
        long weights = (long) (parameters.embeddingDimensions() + parameters.dimensions() + 1) * words
                + 2L * parameters.embeddingDimensions()
                + 4L * parameters.dimensions() * (parameters.embeddingDimensions() + 3L * parameters.dimensions() + 2);

        if (cells > 12_000_000 || weights > 5_000_000) {
            throw new InvalidInputException("LSTM: уменьшите размер пакета, длину, словарь или размерность сети");
        }
        var graph = LstmNetwork.create(parameters, words);
        var diagnostic = LstmNetwork.batch(rows.subList(0, batchSize), words);
        double initialLoss = tokenLoss(graph, diagnostic);
        var random = new Random(parameters.seed());

        for (int epoch = 0; epoch < parameters.epochs(); epoch++) {
            Collections.shuffle(rows, random);

            for (int start = 0; start < rows.size(); start += batchSize) {
                var batch = LstmNetwork.batch(rows.subList(start, Math.min(start + batchSize, rows.size())), words);
                graph.fit(batch);

                if (!Double.isFinite(graph.score())) {
                    throw new ModelTrainingException("LSTM: неконечная потеря при обучении");
                }
            }

            log.info("[LSTM] Эпоха {}/{}", epoch + 1, parameters.epochs());
        }

        return new LstmModel(graph, vocabulary.words(), parameters, initialLoss, tokenLoss(graph, diagnostic));
    }

    private static double tokenLoss(ComputationGraph graph, DataSet batch) {
        double loss = graph.score(batch) * batch.numExamples() / batch.getLabelsMaskArray().sumNumber().doubleValue();

        if (!Double.isFinite(loss)) {
            throw new ModelTrainingException("LSTM: неконечная ошибка восстановления");
        }

        return loss;
    }

    @Override
    protected int workers(LstmParameters parameters) {
        return parameters.workers();
    }

    @Override
    protected int vocabularySize(LstmModel model) {
        return model.vocabulary().size();
    }

    @Override
    protected int dimensions(LstmModel model) {
        return model.parameters().dimensions();
    }
}
