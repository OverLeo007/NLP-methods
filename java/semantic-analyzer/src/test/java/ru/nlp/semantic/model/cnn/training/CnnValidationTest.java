package ru.nlp.semantic.model.cnn.training;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

import org.junit.jupiter.api.Test;

import ru.nlp.semantic.model.cnn.dto.CnnParameters;
import ru.nlp.semantic.model.shared.training.PairSampling;

class CnnValidationTest {

    @Test
    void seededSplitKeepsDuplicateDocumentsTogetherAndPreservesTheirFrequencies() {
        var documents = documents();
        documents.add(documents.getFirst());
        var split = CnnTrainingSplit.create(documents, 0.2, 42);
        assertThat(split).isEqualTo(CnnTrainingSplit.create(documents, 0.2, 42));
        var training = new HashSet<>(split.training());
        assertThat(split.validation()).noneMatch(training::contains);
        assertThat(split.training().size() + split.validation().size()).isEqualTo(documents.size());
        assertThat(
                Collections.frequency(split.training(), documents.getFirst())
                        + Collections.frequency(split.validation(), documents.getFirst())
        ).isEqualTo(2);
        assertThat(CnnTrainingSplit.create(documents, 0, 42).training()).isEqualTo(documents);
        assertThatThrownBy(() -> CnnTrainingSplit.create(List.of(List.of("a"), List.of("b")), 0.1, 42))
                .hasMessageContaining("четыре");
    }

    @Test
    void vocabularyExcludesHeldOutWordsAndValidationHistorySelectsAnEpoch() {
        var documents = documents();
        var split = CnnTrainingSplit.create(documents, 0.2, 42);
        var parameters = CnnParameters.builder()
                .dimensions(8)
                .embeddingDimensions(8)
                .filters(8)
                .kernelSize(3)
                .maxLength(12)
                .maxFeatures(100)
                .minCount(1)
                .batchSize(4)
                .epochs(3)
                .learningRate(0.003)
                .temperature(0.2)
                .seed(42)
                .workers(1)
                .validationFraction(0.2)
                .pairSampling(PairSampling.symmetric)
                .build();
        var model = new CnnTrainer().train(documents, parameters).model();

        for (var heldOut : split.validation()) {
            assertThat(model.vocabulary()).doesNotContain(heldOut.getLast());
        }

        var statistics = model.training();
        assertThat(statistics.trainingDocuments()).isEqualTo(16);
        assertThat(statistics.validationDocuments()).isEqualTo(4);
        assertThat(statistics.validationLosses()).hasSize(3).allSatisfy(value -> assertThat(value).isFinite());
        int expectedEpoch = statistics.validationLosses().indexOf(Collections.min(statistics.validationLosses())) + 1;
        assertThat(statistics.bestValidationEpoch()).isEqualTo(expectedEpoch);
    }

    private static ArrayList<List<String>> documents() {
        String[] words = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta"};
        var documents = new ArrayList<List<String>>();

        for (int i = 0; i < 20; i++) {
            documents.add(List.of(words[i % 6], words[(i + 1) % 6], words[i / 6 + 2], "unique" + i));
        }

        return documents;
    }
}
