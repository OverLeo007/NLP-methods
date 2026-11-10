package ru.nlp.laba3.corpus;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;

class CorpusPreparationTest {

    @TempDir
    Path directory;

    @Test
    void rejectsCorpusContainingOnlyStopwordsAfterNormalization() throws Exception {
        var input = directory.resolve("stopwords.csv");
        Files.writeString(input, "id,text\na,и в на\nb,а от до\n");
        var preparation = new CorpusPreparation(TestComponents.loader(), TestComponents.preprocessor());
        assertThatThrownBy(() -> preparation.prepare(input, 10))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("После нормализации");
    }

    @Test
    void preparationKeepsSourceOrderCountersAndBodyTokensInAnImmutableSnapshot() throws Exception {
        var input = directory.resolve("documents.csv");
        Files.writeString(
                input,
                """
                        id,text,title
                        a,Космос ракеты,Секретный заголовок
                        empty,?!,Пустой
                        duplicate,Космос ракеты,Дубликат
                        b,Футбол команды,Другой заголовок
                        """
        );
        var preprocessor = TestComponents.preprocessor();
        var preparation = new CorpusPreparation(TestComponents.loader(), preprocessor);
        var prepared = preparation.prepare(input, 10);
        assertThat(prepared.corpus().documents()).extracting(Document::id).containsExactly("a", "b");
        assertThat(prepared.corpus().scanned()).isEqualTo(4);
        assertThat(prepared.corpus().empty()).isEqualTo(1);
        assertThat(prepared.corpus().duplicates()).isEqualTo(1);
        assertThat(prepared.documents()).containsExactly(
                preprocessor.tokens("Космос ракеты"),
                preprocessor.tokens("Футбол команды")
        );
        assertThat(prepared.tokens()).isEqualTo(prepared.documents().stream().mapToLong(List::size).sum());
        assertThat(preparation.prepare(input, 10)).isEqualTo(prepared);
        assertThatThrownBy(() -> prepared.documents().clear()).isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> prepared.documents().getFirst().clear())
                .isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> prepared.corpus().documents().clear())
                .isInstanceOf(UnsupportedOperationException.class);
    }
}
