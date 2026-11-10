package ru.nlp.laba3.preprocessing;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;

import org.junit.jupiter.api.Test;

import ru.nlp.laba3.TestComponents;

class TextPreprocessorTest {

    @Test
    void stemsStopwordsInSameSpaceButKeepsNegationsAndNumbers() throws IOException {
        var normalizer = TestComponents.preprocessor();
        assertThat(normalizer.tokens("Ёлками и машины НЕ были ни 12,5"))
                .containsExactly("елк", "машин", "не", "ни", "12,5");
        assertThat(normalizer.tokens("и в на")).isEmpty();
    }

    @Test
    void normalizesHyphenPartsWithoutStemmingLatinOrApostropheWords() throws IOException {
        var normalizer = TestComponents.preprocessor();
        assertThat(normalizer.tokens("Машины-ёлки don't PYTHON"))
                .containsExactly("машин-елк", "don't", "python");
    }
}
