package ru.nlp.laba3.evaluation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import java.util.Set;
import java.util.stream.IntStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.search.SearchHit;
import ru.nlp.laba3.model.search.SearchResult;

class RetrievalMetricsCalculatorTest {

    @ParameterizedTest
    @CsvSource({"1,1.0,1.0", "5,1.0,0.2", "6,0.0,0.16666666666666666", "10,0.0,0.1", "11,0.0,0.0"})
    void honorsBothRankCutoffs(int rank, double recall, double reciprocal) {
        var hits = IntStream.rangeClosed(1, 11)
                .mapToObj(i -> new SearchHit(i == rank ? "relevant" : "d" + i, "", 1.0 / i, "")).toList();
        var scored = RetrievalMetricsCalculator.score(
                new EvaluationQuery("q", "запрос"),
                Set.of("relevant"),
                Set.of(),
                new SearchResult("ok", 1, 1, hits),
                0.1
        );
        assertThat(scored.recallAt5()).isEqualTo(recall);
        assertThat(scored.reciprocalRankAt10()).isCloseTo(reciprocal, within(1e-15));
        assertThat(scored.hits()).hasSize(10);
    }

    @Test
    void computesMacroRecallForMultipleRelevantDocumentsAndKeepsFailedQueries() {
        var query = new EvaluationQuery("q", "запрос");
        var success = RetrievalMetricsCalculator.score(
                query,
                Set.of("a", "b", "c"),
                Set.of("c"),
                new SearchResult("ok", 2, 3, List.of(hit("x"), hit("a"), hit("b"))),
                0.1
        );
        var oov = RetrievalMetricsCalculator.score(
                new EvaluationQuery("oov", "неизвестно"),
                Set.of("a"),
                Set.of(),
                new SearchResult("oov", 0, 1, List.of()),
                0.2
        );
        var empty = RetrievalMetricsCalculator.score(
                new EvaluationQuery("empty", ""),
                Set.of("a"),
                Set.of(),
                new SearchResult("empty_after_normalization", 0, 0, List.of()),
                0.3
        );
        var zero = RetrievalMetricsCalculator.score(
                new EvaluationQuery("zero", "ноль"),
                Set.of("a"),
                Set.of(),
                new SearchResult("zero_vector", 1, 1, List.of()),
                0.4
        );
        var metrics = RetrievalMetricsCalculator.aggregate(List.of(success, oov, empty, zero));
        assertThat(success.recallAt5()).isCloseTo(2.0 / 3, within(1e-15));
        assertThat(success.reciprocalRankAt10()).isEqualTo(0.5);
        assertThat(success.zeroRelevantIds()).containsExactly("c");
        assertThat(success.recallLossReason()).isEqualTo("zero_relevant_document");
        assertThat(metrics.queries()).isEqualTo(4);
        assertThat(metrics.recallAt5()).isCloseTo(1.0 / 6, within(1e-15));
        assertThat(metrics.mrrAt10()).isEqualTo(0.125);
        assertThat(metrics.knownTokens()).isEqualTo(3);
        assertThat(metrics.totalTokens()).isEqualTo(5);
        assertThat(metrics.tokenCoverage()).isEqualTo(0.6);
        assertThat(metrics.partialCoverageQueries()).isEqualTo(1);
        assertThat(metrics.hitQueriesAt5()).isEqualTo(1);
        assertThat(metrics.hitQueriesAt10()).isEqualTo(1);
        assertThat(metrics.queriesWithZeroRelevantDocuments()).isEqualTo(1);
        assertThat(metrics.statuses()).containsEntry("oov", 1L).containsEntry("zero_vector", 1L)
                .containsEntry("empty_after_normalization", 1L).containsEntry("ok", 1L);
        assertThat(metrics.recallLossReasons()).hasSize(4);
        assertThat(metrics.searchSeconds()).isEqualTo(1);
        assertThat(metrics.meanSearchSeconds()).isEqualTo(0.25);
        assertThat(metrics.p95SearchSeconds()).isEqualTo(0.4);
    }

    @Test
    void doesNotInflateRecallForDuplicateHitsAndDistinguishesRankingLoss() {
        var scored = RetrievalMetricsCalculator.score(
                new EvaluationQuery("q", "текст"),
                Set.of("a", "b"),
                Set.of(),
                new SearchResult("ok", 1, 1, List.of(hit("a"), hit("a"))),
                0
        );
        assertThat(scored.recallAt5()).isEqualTo(0.5);
        assertThat(scored.recallLossReason()).isEqualTo("ranking");
        assertThatThrownBy(() -> scored.hits().clear()).isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> scored.relevantIds().clear()).isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void rejectsUndefinedDenominatorsAndDefinesZeroTokenCoverage() {
        assertThatThrownBy(() -> RetrievalMetricsCalculator.aggregate(List.of()))
                .isInstanceOf(InvalidInputException.class);
        assertThatThrownBy(
                () -> RetrievalMetricsCalculator.score(
                        new EvaluationQuery("q", ""),
                        Set.of(),
                        Set.of(),
                        new SearchResult("ok", 0, 0, List.of()),
                        0
                )
        ).isInstanceOf(InvalidInputException.class);
        var empty = RetrievalMetricsCalculator.score(
                new EvaluationQuery("q", ""),
                Set.of("d"),
                Set.of(),
                new SearchResult("empty_after_normalization", 0, 0, List.of()),
                0
        );
        assertThat(RetrievalMetricsCalculator.aggregate(List.of(empty)).tokenCoverage()).isZero();
    }

    private static SearchHit hit(String id) {
        return new SearchHit(id, "", 1, "");
    }
}
