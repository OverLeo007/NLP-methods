package ru.nlp.laba3.corpus;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.springframework.stereotype.Component;

import ru.nlp.laba3.corpus.reader.CorpusReader;
import ru.nlp.laba3.exception.CorpusReadException;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Corpus;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.preprocessing.TextPreprocessor;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Отбор уникальных непустых документов из потока любого поддержанного формата. */
@Component
@Slf4j
@RequiredArgsConstructor
public class CorpusLoader {

    private final CorpusReader reader;

    private final TextPreprocessor preprocessor;

    /**
     * Выбирает первые уникальные непустые документы, не читая оценочные запросы.
     */
    public Corpus read(Path input, int limit) {
        try {
            return readCorpus(input, limit);
        } catch (IOException | UncheckedIOException exception) {
            throw new CorpusReadException(
                    "Не удалось прочитать корпус " + input + ": " + exception.getMessage(),
                    exception
            );
        }
    }

    private Corpus readCorpus(Path input, int limit) throws IOException {

        if (limit < 2) {
            throw new InvalidInputException("Лимит документов должен быть не меньше 2");
        }

        var selection = new Selection(limit, preprocessor);
        log.info("Чтение корпуса: {}, лимит документов={}", input, limit);

        reader.scan(input, document -> {
            selection.add(document);

            return !selection.full();
        });

        if (selection.documents.size() < 2) {
            throw new InvalidInputException("После очистки требуется минимум 2 непустых уникальных документа");
        }

        log.info(
                "Корпус готов: документов={}, просмотрено={}, пустых={}, дубликатов={}",
                selection.documents.size(),
                selection.scanned,
                selection.empty,
                selection.duplicates
        );

        return new Corpus(
                List.copyOf(selection.documents),
                selection.scanned,
                selection.empty,
                selection.duplicates
        );
    }

    @RequiredArgsConstructor
    private static final class Selection {

        private final int limit;

        private final TextPreprocessor preprocessor;

        private final List<Document> documents = new ArrayList<>();

        private final Set<String> ids = new HashSet<>();

        private final Set<String> bodies = new HashSet<>();

        private long scanned;

        private long empty;

        private long duplicates;

        private boolean full() {
            return documents.size() >= limit;
        }

        private void add(Document document) {
            scanned++;

            if (document.id() == null || document.id().isBlank()) {
                throw new InvalidInputException("Пустой ID документа в записи " + scanned);
            }

            if (document.text() == null || !preprocessor.hasTokens(document.text())) {
                empty++;

                return;
            }

            String body = Normalizer.normalize(document.text(), Normalizer.Form.NFC).strip();

            if (ids.contains(document.id()) || bodies.contains(body)) {
                duplicates++;

                return;
            }

            ids.add(document.id());
            bodies.add(body);
            documents.add(document);
        }
    }
}
