package ru.nlp.laba3.corpus.reader;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.model.corpus.Document;

class CorpusReaderContractTest {

    @TempDir
    Path directory;

    @ParameterizedTest
    @EnumSource(Format.class)
    void readsRawDocumentsThroughOneContractWithoutFiltering(Format format) throws Exception {
        var input = source(format, false);
        CorpusReader reader = TestComponents.corpusReader();
        var documents = new ArrayList<Document>();
        assertThat(reader.supports(input)).isTrue();
        assertThat(reader.scan(input, document -> {
            documents.add(document);

            return true;
        })).isEqualTo(3);
        assertThat(documents).containsExactly(
                new Document("a.txt", "a.txt", "Ёлки, РАКЕТЫ\n\"Спутники\""),
                new Document("b.txt", "b.txt", "?!"),
                new Document("c.txt", "c.txt", "Ёлки, РАКЕТЫ\n\"Спутники\"")
        );
    }

    @ParameterizedTest
    @EnumSource(Format.class)
    void stopsBeforeMalformedTailAndReleasesSource(Format format) throws Exception {
        var input = source(format, true);
        CorpusReader reader = TestComponents.corpusReader();
        var documents = new ArrayList<Document>();
        assertThat(reader.scan(input, document -> {
            documents.add(document);

            return false;
        })).isEqualTo(1);
        assertThat(documents).extracting(Document::id).containsExactly("a.txt");

        if (Files.isDirectory(input)) {
            Files.delete(input.resolve("a.txt"));
            Files.delete(input.resolve("b.txt"));
            Files.delete(input.resolve("c.txt"));
        }
        Files.delete(input);
        assertThat(input).doesNotExist();
    }

    private Path source(Format format, boolean malformedTail) throws Exception {
        String body = "Ёлки, РАКЕТЫ\n\"Спутники\"";

        if (format == Format.TXT) {
            var input = Files.createDirectory(directory.resolve("documents.csv"));
            Files.writeString(input.resolve("a.txt"), body);

            if (malformedTail) {
                Files.write(input.resolve("b.txt"), new byte[]{(byte) 0xc3, 0x28});
            } else {
                Files.writeString(input.resolve("b.txt"), "?!");
            }
            Files.writeString(input.resolve("c.txt"), body);

            return input;
        }

        if (format == Format.CSV) {
            var input = directory.resolve("documents.CSV");
            String first = "id,title,text\na.txt,a.txt,\"Ёлки, РАКЕТЫ\n\"\"Спутники\"\"\"\n";
            String tail = malformedTail
                    ? "b.txt,b.txt,\"unfinished"
                    : "b.txt,b.txt,?!\nc.txt,c.txt,\"Ёлки, РАКЕТЫ\n\"\"Спутники\"\"\"\n";
            Files.writeString(input, first + tail);

            return input;
        }
        var mapper = new JsonConfiguration().jsonMapper();
        var lines = new StringBuilder();

        for (String id : List.of("a.txt", "b.txt", "c.txt")) {
            if (malformedTail && id.equals("b.txt")) {
                lines.append("not json\n");
                break;
            }
            lines.append(
                    mapper.writeValueAsString(Map.of("_id", id, "title", id, "text", id.equals("b.txt") ? "?!" : body))
            )
                    .append('\n');
        }

        if (format == Format.JSONL) {
            return Files.writeString(directory.resolve("documents.JSONL"), lines.toString());
        }
        var input = directory.resolve("documents.ZIP");

        try (var zip = new ZipOutputStream(Files.newOutputStream(input), StandardCharsets.UTF_8)) {
            zip.putNextEntry(new ZipEntry("__MACOSX/._corpus.jsonl"));
            zip.write("not json".getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
            zip.putNextEntry(new ZipEntry("corpus.jsonl"));
            zip.write(lines.toString().getBytes(StandardCharsets.UTF_8));
            zip.closeEntry();
        }

        return input;
    }

    private enum Format {
        CSV, TXT, JSONL, ZIP
    }
}
