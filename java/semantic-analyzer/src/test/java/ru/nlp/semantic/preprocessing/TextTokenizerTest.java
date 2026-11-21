package ru.nlp.semantic.preprocessing;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

import ru.nlp.semantic.config.PreprocessingConfiguration;

class TextTokenizerTest {

    @Test
    void preservesWordsConnectorsAndNumbersAfterUnicodeNormalization() {
        assertThat(
                TextTokenizer.tokens(
                        new PreprocessingConfiguration().textNormalizationChain()
                                .normalize("Е\u0308лка, САНКТ-ПЕТЕРБУРГ и 12,5; don't 2026!")
                )
        )
                .containsExactly("елка", "санкт-петербург", "и", "12,5", "don't", "2026");
    }

    @Test
    void blankAndPunctuationHaveNoTokens() {
        assertThat(TextTokenizer.tokens(" \n?!")).isEmpty();
    }
}
