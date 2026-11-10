package ru.nlp.laba3.training;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.runtime.NativeRuntime;
import ru.nlp.laba3.vectorization.TfIdf;

class LsaTrainerTest {

    @Test
    void sublinearTfUsesOnlyKnownTermsBeforeL2Normalization() {
        var vector = TfIdf.sublinearVector(
                List.of("a", "a", "b", "unknown"),
                Map.of("a", 0, "b", 1),
                new double[]{2, 3}
        );
        double a = (1 + Math.log(2)) * 2;
        double norm = Math.sqrt(a * a + 9);
        assertThat(vector).hasSize(2);
        assertThat(vector.get(0)).isCloseTo(a / norm, within(1e-12));
        assertThat(vector.get(1)).isCloseTo(3 / norm, within(1e-12));
    }

    @Test
    void truncatedSvdKeepsLeadingTermSubspaceAndCorpusOnlyIdf() throws Exception {
        var documents = List.of(
                List.of("a"),
                List.of("a", "a"),
                List.of("a", "a", "a"),
                List.of("b"),
                List.of("b", "b"),
                List.of("c")
        );
        var result = NativeRuntime.withThreads(
                1,
                () -> new LsaTrainer().train(documents, new LsaParameters(50, 20000, 1)).model()
        );
        assertThat(result.vocabulary()).containsExactly("a", "b", "c");
        assertThat(result.singularValues()).hasSize(2);
        assertThat(result.singularValues()[0]).isCloseTo(Math.sqrt(3), within(1e-9));
        assertThat(result.singularValues()[1]).isCloseTo(Math.sqrt(2), within(1e-9));
        assertThat(result.idf()[0]).isCloseTo(Math.log(7.0 / 4) + 1, within(1e-12));
        assertThat(result.idf()[1]).isCloseTo(Math.log(7.0 / 3) + 1, within(1e-12));
        assertThat(result.idf()[2]).isCloseTo(Math.log(7.0 / 2) + 1, within(1e-12));

        for (int j = 0; j < 2; j++) {
            double norm = 0;

            for (var row : result.projection()) {
                norm += row[j] * row[j];
            }

            assertThat(norm).isCloseTo(1, within(1e-9));
            assertThat(result.projection()[2][j]).isCloseTo(0, within(1e-9));
        }

        double dot = 0;

        for (var row : result.projection()) {
            dot += row[0] * row[1];
        }

        assertThat(dot).isCloseTo(0, within(1e-9));
    }

    @Test
    void rectangularSvdHandlesMoreThanSixteenSingularValues() throws Exception {
        var documents = new java.util.ArrayList<List<String>>();

        for (int term = 0; term < 20; term++) {
            for (int copy = 0; copy <= term; copy++) {
                documents.add(List.of("term" + term));
            }
        }

        var result = NativeRuntime.withThreads(
                1,
                () -> new LsaTrainer().train(documents, new LsaParameters(50, 20000, 1)).model()
        );
        assertThat(result.singularValues()).hasSize(19);

        for (int i = 0; i < 19; i++) {
            assertThat(result.singularValues()[i]).isCloseTo(Math.sqrt(20 - i), within(1e-9));
        }
    }

    @Test
    void rejectsInsufficientDataAndBoundsDenseAllocation() {
        var trainer = new LsaTrainer();
        var parameters = new LsaParameters(2, 20000, 1);
        assertThatThrownBy(() -> trainer.train(List.of(List.of("a", "b")), parameters))
                .hasMessageContaining("минимум 2 документа");
        assertThatThrownBy(() -> trainer.train(List.of(List.of("a"), List.of("a")), parameters))
                .hasMessageContaining("2 различных");
        var terms = java.util.stream.IntStream.range(0, 20000).mapToObj(Integer::toString).toList();
        var large = java.util.stream.IntStream.range(0, 1001)
                .mapToObj(i -> i == 0 ? terms : List.of("0")).toList();
        assertThatThrownBy(() -> trainer.train(large, parameters)).hasMessageContaining("20000000 ячеек");
    }
}
