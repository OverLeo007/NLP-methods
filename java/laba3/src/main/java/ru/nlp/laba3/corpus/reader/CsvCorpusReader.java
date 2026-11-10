package ru.nlp.laba3.corpus.reader;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.MappingIterator;
import com.fasterxml.jackson.dataformat.csv.CsvMapper;
import com.fasterxml.jackson.dataformat.csv.CsvSchema;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;

import lombok.RequiredArgsConstructor;

/** Читает UTF-8 CSV с заголовками id,text и необязательным title. */
@Component
@RequiredArgsConstructor
public class CsvCorpusReader extends AbstractCorpusReader {

    private final CsvMapper csvMapper;

    /** Распознаёт CSV независимо от регистра расширения. */
    @Override
    public boolean supports(Path input) {
        return !Files.isDirectory(input) && input.toString().toLowerCase(Locale.ROOT).endsWith(".csv");
    }

    /** Открывает CSV; парсер создаётся при первом чтении под управлением общего цикла. */
    @Override
    protected DocumentCursor open(Path input) throws IOException {
        return new CsvCursor(Files.newBufferedReader(input, StandardCharsets.UTF_8));
    }

    @RequiredArgsConstructor
    private final class CsvCursor implements DocumentCursor {

        private final BufferedReader reader;

        private MappingIterator<Map<String, String>> rows;

        @Override
        public Document next() throws IOException {
            if (rows == null) {
                rows = csvMapper.readerFor(Map.class).with(CsvSchema.emptySchema().withHeader()).readValues(reader);
            }

            if (!rows.hasNextValue()) {
                return null;
            }
            var row = rows.nextValue();

            if (!row.containsKey("id") || !row.containsKey("text")) {
                throw new InvalidInputException("CSV должен содержать заголовки id,text");
            }

            return new Document(row.get("id"), row.getOrDefault("title", ""), row.get("text"));
        }

        @Override
        public void close() throws IOException {
            try (reader) {
                if (rows != null) {
                    rows.close();
                }
            }
        }
    }
}
