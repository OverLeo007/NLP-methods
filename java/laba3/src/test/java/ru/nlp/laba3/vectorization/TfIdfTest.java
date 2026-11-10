package ru.nlp.laba3.vectorization;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.List;

import org.junit.jupiter.api.Test;

class TfIdfTest {

    @Test
    void idfCountsDocumentsRatherThanOccurrencesAndNeverFitsQueryWords() {
        var idf = TfIdf.fitIdf(List.of(List.of("космос", "космос", "ракет"), List.of("космос")));
        assertThat(idf.get("космос")).isEqualTo(1.0);
        assertThat(idf.get("ракет")).isCloseTo(Math.log(3.0 / 2) + 1, within(1e-12));
        assertThat(idf).doesNotContainKey("невиданное");
    }
}
