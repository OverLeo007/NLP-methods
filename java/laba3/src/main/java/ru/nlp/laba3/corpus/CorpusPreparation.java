package ru.nlp.laba3.corpus;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import org.springframework.stereotype.Component;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.PreparedCorpus;
import ru.nlp.laba3.preprocessing.TextPreprocessor;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Единая подготовка корпуса для всех анализаторов без обучения на запросах. */
@Component
@RequiredArgsConstructor
@Slf4j
public class CorpusPreparation {

    private final CorpusLoader loader;

    private final TextPreprocessor preprocessor;

    /** Читает выбранную часть источника и нормализует только тела документов. */
    public PreparedCorpus prepare(Path input, int limit) throws IOException {
        var corpus = loader.read(input, limit);
        var documents = corpus.documents().stream().map(document -> preprocessor.tokens(document.text())).toList();
        long tokens = documents.stream().mapToLong(List::size).sum();

        if (tokens == 0) {
            throw new InvalidInputException("После нормализации в корпусе нет слов");
        }
        log.info("Нормализация завершена: документов={}, токенов={}", corpus.documents().size(), tokens);

        return new PreparedCorpus(corpus, documents, tokens);
    }
}
