package ru.nlp.laba3.corpus;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.exception.CorpusReadException;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;

class CorpusLoaderTest {

    @TempDir
    Path directory;

    @Test
    void readsQuotedMultilineCsvAndReportsSkippedRows() throws Exception {
        Path csv = directory.resolve("documents.csv");
        Files.writeString(
                csv,
                """
                        id,text,title
                        1,"Космос, ракеты
                        Спутники",Заголовок
                        2,?!,Пустой
                        3,Футбол матчи,Спорт
                        4,Футбол матчи,Дубликат
                        """
        );
        var corpus = TestComponents.loader().read(csv, 10);
        assertThat(corpus.documents()).extracting(Document::id).containsExactly("1", "3");
        assertThat(corpus.documents().getFirst().text())
                .contains("Космос, ракеты\nСпутники")
                .doesNotContain("Заголовок");
        assertThat(corpus.scanned()).isEqualTo(4);
        assertThat(corpus.empty()).isEqualTo(1);
        assertThat(corpus.duplicates()).isEqualTo(1);
    }

    @Test
    void txtOrderAndLimitAreDeterministic() throws Exception {
        Files.writeString(directory.resolve("b.txt"), "Футбол матчи");
        Files.writeString(directory.resolve("a.txt"), "Космос ракеты");
        Files.writeString(directory.resolve("c.txt"), "Музыка концерты");
        var corpus = TestComponents.loader().read(directory, 2);
        assertThat(corpus.documents()).extracting(Document::id).containsExactly("a.txt", "b.txt");
    }

    @Test
    void rejectsBlankIdsAndInsufficientUniqueDocuments() throws Exception {
        var input = directory.resolve("invalid.csv");
        Files.writeString(input, "id,text\n,Космос ракеты\na,Футбол команды\n");
        assertThatThrownBy(() -> TestComponents.loader().read(input, 10))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Пустой ID");
        Files.writeString(input, "id,text\na,Космос ракеты\nb,Космос ракеты\nc,?!\n");
        assertThatThrownBy(() -> TestComponents.loader().read(input, 10))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("минимум 2");
    }

    @Test
    void missingCorpusPreservesCauseWhileValidationDoesNotCaptureStack() throws Exception {
        var reader = TestComponents.loader();
        assertThatThrownBy(() -> reader.read(directory.resolve("missing.csv"), 2))
                .isInstanceOf(CorpusReadException.class)
                .hasCauseInstanceOf(IOException.class);

        try {
            reader.read(directory, 1);
            throw new AssertionError("Ожидалась ошибка лимита");
        } catch (InvalidInputException exception) {
            assertThat(exception.getStackTrace()).isEmpty();
            exception.fillInStackTrace();
            assertThat(exception.getStackTrace()).isEmpty();
            assertThat(exception.getMessage()).contains("Лимит");
        }
    }
}
