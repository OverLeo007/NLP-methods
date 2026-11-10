package ru.nlp.laba3.vectorization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class LsaVectorizerTest {

    @Test
    void usesFittedTransformAndCountsOccurrencesWithoutExposingItsArrays() {
        var vocabulary = new ArrayList<>(List.of("a", "b"));
        double[] idf = {2, 3};
        double[][] projection = {{1, 0}, {0, 1}};
        TextVectorizer vectorizer = new LsaVectorizer(vocabulary, idf, projection);
        vocabulary.clear();
        idf[0] = 100;
        projection[0][0] = 0;
        var encoded = vectorizer.encode(List.of("a", "a", "b", "unknown"));
        double a = (1 + Math.log(2)) * 2;
        double norm = Math.sqrt(a * a + 9);
        assertThat(encoded.knownTokens()).isEqualTo(3);
        assertThat(vectorizer.dimensions()).isEqualTo(2);
        assertThat(encoded.values()[0]).isCloseTo(a / norm, within(1e-12));
        assertThat(encoded.values()[1]).isCloseTo(3 / norm, within(1e-12));
        double[] published = encoded.values();
        published[0] = 100;
        assertThat(encoded.values()[0]).isCloseTo(a / norm, within(1e-12));
        assertThat(vectorizer.encode(List.of("unknown")).isZero()).isTrue();
        assertThat(vectorizer.encode(List.of("unknown")).knownTokens()).isZero();
    }

    @Test
    void retainsLsaNumericalZeroThresholdWithoutChangingCoverage() {
        var vectorizer = new LsaVectorizer(
                List.of("a", "b"),
                new double[]{1, 1},
                new double[][]{{1e-13}, {1}}
        );
        var zero = vectorizer.encode(List.of("a"));
        assertThat(zero.isZero()).isTrue();
        assertThat(zero.knownTokens()).isEqualTo(1);
        assertThat(vectorizer.encode(List.of("b")).values()).containsExactly(1);
        double[] input = {1};
        var encoded = new EncodedText(input, 1);
        input[0] = 0;
        assertThat(encoded.values()).containsExactly(1);
    }
}
