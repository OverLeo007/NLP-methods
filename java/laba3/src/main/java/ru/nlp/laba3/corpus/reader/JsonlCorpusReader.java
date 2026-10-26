package ru.nlp.laba3.corpus.reader;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.zip.ZipFile;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;

import lombok.RequiredArgsConstructor;

/** Потоковый источник документов JSONL, общий для обучения и подготовки выборки. */
@Component
@RequiredArgsConstructor
public class JsonlCorpusReader extends AbstractCorpusReader {

    private final ObjectMapper jsonMapper;

    /** Распознаёт JSONL и ZIP РИА независимо от регистра расширения. */
    @Override
    public boolean supports(Path input) {
        String name = input.toString().toLowerCase(Locale.ROOT);

        return !Files.isDirectory(input) && (name.endsWith(".jsonl") || name.endsWith(".zip"));
    }

    /** Передаёт файлы общему циклу; архивная запись открывается при первом чтении. */
    @Override
    protected DocumentCursor open(Path input) throws IOException {
        return new JsonlCursor(input);
    }

    private final class JsonlCursor implements DocumentCursor {

        private BufferedReader reader;

        private final ZipFile archive;

        private long lineNumber;

        private JsonlCursor(Path input) throws IOException {
            if (input.toString().toLowerCase(Locale.ROOT).endsWith(".zip")) {
                archive = new ZipFile(input.toFile(), StandardCharsets.UTF_8);
            } else {
                archive = null;
                reader = Files.newBufferedReader(input, StandardCharsets.UTF_8);
            }
        }

        @Override
        public Document next() throws IOException {
            if (reader == null) {
                reader = openArchiveEntry();
            }
            String line;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                if (line.isBlank()) {
                    continue;
                }
                var row = jsonMapper.readTree(line);

                if (!row.hasNonNull("_id") || !row.path("text").isTextual()) {
                    throw new InvalidInputException("JSONL: обязательны _id и строковый text, строка " + lineNumber);
                }

                return new Document(row.get("_id").asText(), row.path("title").asText(""), row.get("text").asText());
            }

            return null;
        }

        @Override
        public void close() throws IOException {
            try (archive) {
                if (reader != null) {
                    reader.close();
                }
            }
        }

        private BufferedReader openArchiveEntry() throws IOException {
            var entry = archive.getEntry("corpus.jsonl");

            if (entry == null) {
                throw new InvalidInputException("В ZIP отсутствует corpus.jsonl");
            }

            return new BufferedReader(new InputStreamReader(archive.getInputStream(entry), StandardCharsets.UTF_8));
        }
    }
}
