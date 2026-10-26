package ru.nlp.laba3.training;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;

import org.junit.jupiter.api.Test;

import ru.nlp.laba3.exception.ModelTrainingException;
import ru.nlp.laba3.model.analysis.Algorithm;

class AbstractModelTrainerTest {

    @Test
    void unexpectedLibraryFailureRetainsCauseAndAlgorithm() {
        var cause = new IllegalStateException("library failure");
        var trainer = new FailingTrainer(Algorithm.word2vec, cause);
        assertThatThrownBy(() -> trainer.train(List.of(List.of("token")), 1))
                .isInstanceOf(ModelTrainingException.class)
                .hasMessageContaining("Word2Vec").hasMessageContaining("library failure").hasCause(cause);
    }

    @Test
    void diagnosedTrainingFailureIsNotWrappedAgain() {
        var cause = new IllegalStateException("native failure");
        var diagnosed = new ModelTrainingException("LSA: SVD failure", cause);
        var trainer = new FailingTrainer(Algorithm.lsa, diagnosed);
        assertThatThrownBy(() -> trainer.train(List.of(List.of("token")), 1)).isSameAs(diagnosed).hasCause(cause);
    }

    private static final class FailingTrainer extends AbstractModelTrainer<Integer, String> {

        private final RuntimeException failure;

        private FailingTrainer(Algorithm algorithm, RuntimeException failure) {
            super(algorithm);
            this.failure = failure;
        }

        @Override
        protected String fit(List<List<String>> documents, Integer parameters) {
            throw failure;
        }

        @Override
        protected int workers(Integer parameters) {
            return parameters;
        }

        @Override
        protected int vocabularySize(String model) {
            return model.length();
        }

        @Override
        protected int dimensions(String model) {
            return model.length();
        }
    }
}
