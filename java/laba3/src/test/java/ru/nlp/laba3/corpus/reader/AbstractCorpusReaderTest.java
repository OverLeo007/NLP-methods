package ru.nlp.laba3.corpus.reader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;

import ru.nlp.laba3.model.corpus.Document;

class AbstractCorpusReaderTest {

    @Test
    void closesCursorAfterEarlyStopWithoutReadingNextRecord() throws Exception {
        var reader = new TrackingReader(false, false);
        assertThat(reader.scan(Path.of("source"), document -> false)).isEqualTo(1);
        assertThat(reader.calls).isEqualTo(1);
        assertThat(reader.closed).isTrue();
    }

    @Test
    void closesCursorWhenConsumerThrowsAndPreservesOriginalFailure() {
        var reader = new TrackingReader(false, false);
        var failure = new IllegalStateException("Ошибка отбора");
        assertThatThrownBy(() -> reader.scan(Path.of("source"), document -> {
            throw failure;
        })).isSameAs(failure);
        assertThat(reader.closed).isTrue();
        assertThat(reader.calls).isEqualTo(1);
    }

    @Test
    void retainsReadFailureAndSuppressesCloseFailure() {
        var reader = new TrackingReader(true, true);
        assertThatThrownBy(() -> reader.scan(Path.of("source"), document -> true))
                .isInstanceOf(IOException.class).hasMessage("Ошибка чтения")
                .satisfies(
                        error -> assertThat(error.getSuppressed()).extracting(Throwable::getMessage)
                                .containsExactly("Ошибка закрытия")
                );
        assertThat(reader.closed).isTrue();
    }

    private static final class TrackingReader extends AbstractCorpusReader {

        private final boolean failRead;

        private final boolean failClose;

        private boolean closed;

        private int calls;

        private TrackingReader(boolean failRead, boolean failClose) {
            this.failRead = failRead;
            this.failClose = failClose;
        }

        @Override
        public boolean supports(Path input) {
            return true;
        }

        @Override
        protected DocumentCursor open(Path input) {
            return new DocumentCursor() {

                @Override
                public Document next() throws IOException {
                    calls++;

                    if (failRead) {
                        throw new IOException("Ошибка чтения");
                    }

                    return new Document("id", "", "Ракеты");
                }

                @Override
                public void close() throws IOException {
                    closed = true;

                    if (failClose) {
                        throw new IOException("Ошибка закрытия");
                    }
                }
            };
        }
    }
}
