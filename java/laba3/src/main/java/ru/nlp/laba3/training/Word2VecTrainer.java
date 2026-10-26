package ru.nlp.laba3.training;

import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.springframework.stereotype.Component;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.word2vec.Word2VecModel;
import ru.nlp.laba3.model.word2vec.Word2VecParameters;
import ru.nlp.laba3.runtime.StageTimings;
import ru.nlp.laba3.vectorization.TfIdf;

import lombok.extern.slf4j.Slf4j;

/** Библиотечное обучение и прогресс; параметры и манифест используют одни константы. */
@Component
@Slf4j
public class Word2VecTrainer extends AbstractModelTrainer<Word2VecParameters, Word2VecModel> {

    private static final int ITERATIONS_PER_EPOCH = 1;

    private static final int NEGATIVE_SAMPLES = 5;

    private static final double LEARNING_RATE = 0.025;

    private static final double MIN_LEARNING_RATE = 0.0001;

    private static final double SAMPLING = 0;

    private static final boolean HIERARCHICAL_SOFTMAX = false;

    private static final boolean PARALLEL_TOKENIZATION = false;

    private static final int HEARTBEAT_SECONDS = 5;

    /** Задаёт алгоритм для общего сценария нативного обучения. */
    public Word2VecTrainer() {
        super(Algorithm.word2vec);
    }

    /** Возвращает число потоков обучения Word2Vec. */
    @Override
    protected int workers(Word2VecParameters parameters) {
        return parameters.workers();
    }

    /** Возвращает фактический размер словаря Word2Vec. */
    @Override
    protected int vocabularySize(Word2VecModel model) {
        return model.vectors().vocab().numWords();
    }

    /** Возвращает фактическую размерность Word2Vec. */
    @Override
    protected int dimensions(Word2VecModel model) {
        return model.parameters().dimensions();
    }

    /** Обучает специфичное состояние Word2Vec внутри общего нативного контекста. */
    @Override
    protected Word2VecModel fit(List<List<String>> documents, Word2VecParameters parameters) {

        validateVocabulary(documents, parameters.minCount());
        long started = System.nanoTime();
        var idf = TfIdf.fitIdf(documents);
        var completedEpochs = new AtomicInteger();
        var model = createModel(documents, parameters, epochListener(parameters.epochs(), started, completedEpochs));
        fitModel(model, documents, parameters, started, completedEpochs);

        if (model.vocab().numWords() == 0) {
            throw new InvalidInputException("Word2Vec: пустой словарь; уменьшите min-count");
        }

        return new Word2VecModel(model, idf, parameters, settings());
    }

    private void fitModel(
            Word2Vec model,
            List<List<String>> documents,
            Word2VecParameters parameters,
            long started,
            AtomicInteger completedEpochs
    ) {
        long words = documents.stream().mapToLong(List::size).sum();
        log.info(
                "[{}] Skip-gram: начало словаря/обучения, эпох={}, workers={}, токенов={}",
                algorithm.displayName(),
                parameters.epochs(),
                parameters.workers(),
                words
        );

        try (var heartbeat = Executors.newSingleThreadScheduledExecutor(runnable -> {
            var thread = new Thread(runnable, algorithm.name() + "-progress");
            thread.setDaemon(true);

            return thread;
        })) {
            heartbeat.scheduleAtFixedRate(
                    () -> log.info(
                            "[{}] Обучение активно: завершено эпох={}/{}, документов={}, токенов={}, прошло={} с",
                            algorithm.displayName(),
                            completedEpochs.get(),
                            parameters.epochs(),
                            documents.size(),
                            words,
                            seconds(started)
                    ),
                    HEARTBEAT_SECONDS,
                    HEARTBEAT_SECONDS,
                    TimeUnit.SECONDS
            );

            try {
                model.fit();
            } finally {
                heartbeat.shutdownNow();
            }
        }
    }

    /** Описывает те же параметры, которые применены в createModel, без дублирования значений. */
    private Map<String, String> settings() {
        return Map.of(
                "algorithm",
                "skip-gram",
                "iterations",
                Integer.toString(ITERATIONS_PER_EPOCH),
                "negativeSamples",
                Integer.toString(NEGATIVE_SAMPLES),
                "learningRate",
                Double.toString(LEARNING_RATE),
                "minLearningRate",
                Double.toString(MIN_LEARNING_RATE),
                "sampling",
                Double.toString(SAMPLING),
                "hierarchicSoftmax",
                Boolean.toString(HIERARCHICAL_SOFTMAX),
                "parallelTokenization",
                Boolean.toString(PARALLEL_TOKENIZATION),
                "weighting",
                "linearTf-smoothedIdf-l2"
        );
    }

    private Word2Vec createModel(
            List<List<String>> documents,
            Word2VecParameters parameters,
            VectorsListener<VocabWord> listener
    ) {
        return new Word2Vec.Builder().layerSize(parameters.dimensions()).epochs(parameters.epochs())
                .iterations(ITERATIONS_PER_EPOCH).minWordFrequency(parameters.minCount())
                .windowSize(parameters.window()).seed(parameters.seed()).workers(parameters.workers())
                .learningRate(LEARNING_RATE).minLearningRate(MIN_LEARNING_RATE).negativeSample(NEGATIVE_SAMPLES)
                .sampling(SAMPLING).useHierarchicSoftmax(HIERARCHICAL_SOFTMAX)
                .elementsLearningAlgorithm(new SkipGram<>()).allowParallelTokenization(PARALLEL_TOKENIZATION)
                .iterate(
                        new CollectionSentenceIterator(
                                documents.stream().map(tokens -> String.join(" ", tokens)).toList()
                        )
                )
                .tokenizerFactory(new DefaultTokenizerFactory()).setVectorsListeners(List.of(listener)).build();
    }

    private VectorsListener<VocabWord> epochListener(int epochs, long started, AtomicInteger completedEpochs) {
        return new VectorsListener<>() {

            @Override
            public boolean validateEvent(ListenerEvent event, long argument) {
                return event == ListenerEvent.EPOCH;
            }

            @Override
            public void processEvent(ListenerEvent event, SequenceVectors<VocabWord> vectors, long argument) {
                log.info(
                        "[{}] Эпоха {}/{} завершена; прошло {} с",
                        algorithm.displayName(),
                        completedEpochs.incrementAndGet(),
                        epochs,
                        seconds(started)
                );
            }
        };
    }

    private void validateVocabulary(List<List<String>> documents, int minCount) {
        var frequencies = new HashMap<String, Long>();
        documents.forEach(tokens -> tokens.forEach(token -> frequencies.merge(token, 1L, Long::sum)));

        if (frequencies.values().stream().filter(frequency -> frequency >= minCount).count() < 2) {
            throw new InvalidInputException(
                    "Word2Vec: после min-count требуется минимум 2 разных слова; уменьшите min-count"
            );
        }
    }

    private static String seconds(long started) {
        return String.format(Locale.ROOT, "%.1f", StageTimings.elapsed(started));
    }

}
