package ru.nlp.laba3.corpus.reader;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.exception.InvalidInputException;

class CorpusReaderValidationTest {

    @TempDir
    Path directory;

    @Test
    void rejectsCsvWithoutRequiredHeadersAndClosesFile() throws Exception {
        var input = Files.writeString(directory.resolve("wrong.csv"), "identifier,body\na,Ракеты спутники\n");
        assertThatThrownBy(() -> TestComponents.corpusReader().scan(input, document -> true))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("id,text");
        Files.delete(input);
        assertThat(input).doesNotExist();
    }

    @Test
    void jsonlValidationUsesPhysicalLineNumberIncludingEmptyLines() throws Exception {
        var input = Files.writeString(directory.resolve("wrong.jsonl"), "\n\n{\"_id\":\"a\"}\n");
        assertThatThrownBy(() -> TestComponents.corpusReader().scan(input, document -> true))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("строка 3");
        Files.delete(input);
    }

    @Test
    void rejectsZipWithoutMainCorpusAndClosesArchive() throws Exception {
        var input = directory.resolve("wrong.zip");

        try (var zip = new ZipOutputStream(Files.newOutputStream(input), StandardCharsets.UTF_8)) {
            zip.putNextEntry(new ZipEntry("__MACOSX/._corpus.jsonl"));
            zip.write("not a corpus".getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }

        assertThatThrownBy(() -> TestComponents.corpusReader().scan(input, document -> true))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("отсутствует corpus.jsonl");
        Files.delete(input);
        assertThat(input).doesNotExist();
    }

    @Test
    void validatesSupportedFormatsBeforeTryingToOpenSource() {
        var input = directory.resolve("missing.unknown");
        var reader = TestComponents.corpusReader();
        assertThat(reader.supports(input)).isFalse();
        assertThatThrownBy(() -> reader.scan(input, document -> true))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Ожидается CSV");
        var csv = new CsvCorpusReader(new JsonConfiguration().csvMapper());
        assertThatThrownBy(() -> csv.scan(input, document -> true))
                .isInstanceOf(InvalidInputException.class).hasMessageContaining("Неподдерживаемый формат");
    }
}
