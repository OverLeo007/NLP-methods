package ru.nlp.semantic.model.shared.training;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.Random;

import org.junit.jupiter.api.Test;

import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;

class PairSamplingTest {

    @Test
    void symmetricBatchesExactlyMatchFingerprintRecordedBeforeExtraction() throws Exception {
        var documents = new ArrayList<int[]>();

        for (int i = 0; i < 40; i++) {
            int[] row = new int[20 + i * 4];

            for (int j = 0; j < row.length; j++) {
                row[j] = 2 + i * 200 + j;
            }

            documents.add(row);
        }

        var pairs = new DocumentPairs(documents, 128, 16);
        var random = new Random(42);
        var hash = MessageDigest.getInstance("SHA-256");
        int batches = 0;

        for (int epoch = 0; epoch < 9; epoch++) {
            for (var batch : pairs.batches(random)) {
                batches++;

                for (int i = 0; i < batch.rows().size(); i++) {
                    hash.update(ByteBuffer.allocate(4).putInt(batch.sources().get(i)).array());
                    hash.update(ByteBuffer.allocate(4).putInt(batch.rows().get(i).length).array());

                    for (int id : batch.rows().get(i)) {
                        hash.update(ByteBuffer.allocate(4).putInt(id).array());
                    }
                }
            }
        }

        assertThat(batches).isEqualTo(27);
        assertThat(HexFormat.of().formatHex(hash.digest()))
                .isEqualTo("e8f7cf9f12d9778190c8ce968c9dadb0f5614b00a1901695c22c26c18b38dea1");
    }

    @Test
    void shortLongSamplingPreservesPositionsLengthBoundsKnownTokensAndSeed() {
        int[] document = new int[256];

        for (int i = 0; i < document.length; i++) {
            document[i] = i + 2;
        }

        var first = new Random(42);
        var second = new Random(42);

        for (int i = 0; i < 100; i++) {
            var pair = PairSampling.short_long.sample(document, 128, first);
            var repeated = PairSampling.short_long.sample(document, 128, second);
            assertThat(pair.first()).containsExactly(repeated.first());
            assertThat(pair.second()).containsExactly(repeated.second());
            assertThat(pair.first().length).isBetween(4, 16);
            assertThat(pair.second().length).isBetween(64, 128);
            assertThat(document).containsSequence(pair.first());
            assertThat(document).containsSequence(pair.second());
        }

        var shortDocument = PairSampling.short_long.sample(new int[]{2}, 128, new Random(42));
        assertThat(shortDocument.first()).containsExactly(2);
        assertThat(shortDocument.second()).containsExactly(2);
        int[] rareKnown = new int[100];
        java.util.Arrays.fill(rareKnown, 1);
        rareKnown[99] = 2;
        var sparse = PairSampling.short_long.sample(rareKnown, 128, new Random(42));
        assertThat(SequenceVocabulary.knownTokens(sparse.first())).isPositive();
        assertThat(SequenceVocabulary.knownTokens(sparse.second())).isPositive();
    }
}
