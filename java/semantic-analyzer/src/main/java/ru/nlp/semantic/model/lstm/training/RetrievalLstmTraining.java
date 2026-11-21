package ru.nlp.semantic.model.lstm.training;

import java.util.List;
import java.util.Random;
import java.util.Set;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.DataSet;

import ru.nlp.semantic.exception.InvalidInputException;
import ru.nlp.semantic.exception.ModelTrainingException;
import ru.nlp.semantic.model.lstm.LstmModel;
import ru.nlp.semantic.model.lstm.dto.LstmParameters;
import ru.nlp.semantic.model.lstm.network.LstmNetwork;
import ru.nlp.semantic.model.shared.training.DocumentPairs;
import ru.nlp.semantic.model.shared.training.PairSampling;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;
import ru.nlp.semantic.model.word2vec.dto.Word2VecParameters;
import ru.nlp.semantic.model.word2vec.training.Word2VecTrainer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Контрастивное обучение одного LSTM-кодировщика для запросов и документов. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
@Slf4j
final class RetrievalLstmTraining {

    static LstmModel fit(List<List<String>> documents, LstmParameters parameters, Word2VecTrainer word2Vec) {
        var vocabulary = SequenceVocabulary.fit(documents, parameters.minCount(), parameters.maxFeatures());
        int dimensions = parameters.dimensions();
        int embedding = parameters.embeddingDimensions();
        long weights = (vocabulary.words().size() + 2L) * embedding + 4L * dimensions * (embedding + dimensions + 1);
        long cells = 2L * Math.min(parameters.batchSize(), documents.size()) * parameters.maxLength()
                * Math.max(embedding, 4 * dimensions);

        if (weights > 5_000_000 || cells > 12_000_000) {
            throw new InvalidInputException(
                    "LSTM retrieval: уменьшите размер пакета, длину, словарь или размерность сети"
            );
        }
        var pairs = new DocumentPairs(
                documents.stream().map(vocabulary::encode).toList(),
                parameters.maxLength(),
                parameters.batchSize(),
                PairSampling.short_long
        );
        var diagnostic = dataSet(pairs.batches(new Random(parameters.seed() ^ 0x5DEECE66DL)).getFirst());
        var graph = LstmNetwork.create(parameters, vocabulary.words().size());
        initializeEmbeddings(graph, documents, vocabulary.words(), parameters, word2Vec);
        double initial = pairLoss(graph, diagnostic);
        var random = new Random(parameters.seed());

        for (int epoch = 0; epoch < parameters.epochs(); epoch++) {
            for (var batch : pairs.batches(random)) {
                graph.fit(dataSet(batch));

                if (!Double.isFinite(graph.score())) {
                    throw new ModelTrainingException("LSTM retrieval: неконечная потеря обучения");
                }
            }

            log.info("[LSTM] Контрастивное обучение: эпоха {}/{}", epoch + 1, parameters.epochs());
        }

        return new LstmModel(graph, vocabulary.words(), parameters, initial, pairLoss(graph, diagnostic));
    }

    private static DataSet dataSet(DocumentPairs.Batch batch) {
        var data = batch.dataSet();
        var ids = data.getFeatures();
        data.setFeatures(ids.reshape(ids.size(0), 1, ids.size(1)));

        return data;
    }

    private static void initializeEmbeddings(
            ComputationGraph graph,
            List<List<String>> documents,
            List<String> words,
            LstmParameters parameters,
            Word2VecTrainer word2Vec
    ) {
        var allowed = Set.copyOf(words);
        var corpus = documents.stream().map(row -> row.stream().filter(allowed::contains).toList())
                .filter(row -> !row.isEmpty()).toList();
        var training = Word2VecParameters.builder().dimensions(parameters.embeddingDimensions()).epochs(5).minCount(1)
                .window(5).seed(parameters.seed()).workers(parameters.workers()).build();
        var initial = word2Vec.fitEmbeddings(corpus, training);
        var table = graph.getLayer("embedding").getParam("W");
        // Строки обеих таблиц представляют их собственную память; putRow копирует значения в граф.
        table.getRow(0).assign(0);
        table.getRow(1).assign(0);

        for (int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            table.putRow(i + 2L, initial.getWordVectorMatrix(word));
        }
    }

    private static double pairLoss(ComputationGraph graph, DataSet batch) {
        double score = graph.score(batch);

        if (!Double.isFinite(score)) {
            throw new ModelTrainingException("LSTM retrieval: неконечная потеря на фиксированных парах");
        }

        return score;
    }
}
