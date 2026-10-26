package ru.nlp.laba3.corpus.reader;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.function.Predicate;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;

/** Общий цикл чтения, ранняя остановка, подсчёт записей и закрытие источника. */
public abstract class AbstractCorpusReader implements CorpusReader {

    /** Выполняет общий контракт независимо от формата и причины остановки. */
    @Override
    public final long scan(Path input, Predicate<Document> consumer) throws IOException {
        Objects.requireNonNull(consumer, "Обработчик документов обязателен");

        if (!supports(input)) {
            throw new InvalidInputException("Неподдерживаемый формат источника: " + input);
        }

        try (var cursor = open(input)) {
            long count = 0;
            Document document;

            while ((document = cursor.next()) != null) {
                count++;

                if (!consumer.test(document)) {
                    break;
                }
            }

            return count;
        } catch (UncheckedIOException exception) {
            throw exception.getCause();
        }
    }

    /** Открывает источник; возвращённый курсор принадлежит общему циклу чтения. */
    protected abstract DocumentCursor open(Path input) throws IOException;

    /** Минимальный внутренний контракт формата с явным владением ресурсами. */
    protected interface DocumentCursor extends AutoCloseable {

        /** Читает очередной документ или возвращает null в конце источника. */
        Document next() throws IOException;

        /** Освобождает файлы источника, включая незавершённое чтение. */
        @Override
        void close() throws IOException;
    }
}
