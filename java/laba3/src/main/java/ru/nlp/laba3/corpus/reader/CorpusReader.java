package ru.nlp.laba3.corpus.reader;

import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Predicate;

import ru.nlp.laba3.model.corpus.Document;

/** Контракт потокового чтения исходных документов без очистки, нормализации и отбора. */
public interface CorpusReader {

    /** Проверяет формат пути без чтения содержимого источника. */
    boolean supports(Path input);

    /**
     * Передаёт исходные документы по одному и возвращает число переданных записей. После false от обработчика не читает
     * следующую запись; закрывает ресурсы также при ошибке.
     */
    long scan(Path input, Predicate<Document> consumer) throws IOException;
}
