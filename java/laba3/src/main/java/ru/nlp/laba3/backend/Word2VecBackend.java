package ru.nlp.laba3.backend;

import java.io.IOException;
import java.nio.file.Path;

import org.springframework.stereotype.Component;

import ru.nlp.laba3.model.search.OpenedModel;
import ru.nlp.laba3.model.training.TrainingSummary;
import ru.nlp.laba3.model.word2vec.Word2VecParameters;
import ru.nlp.laba3.search.SearchSession;
import ru.nlp.laba3.service.Word2VecService;

import lombok.RequiredArgsConstructor;

/** Адаптер фронтенда к сценариям сервиса Word2Vec. */
@Component
@RequiredArgsConstructor
public class Word2VecBackend implements ModelBackend<Word2VecParameters> {

    private final Word2VecService service;

    /** Передаёт обучение и сохранение сервису Word2Vec. */
    @Override
    public TrainingSummary train(Path input, Path output, int limit, Word2VecParameters parameters) throws IOException {
        return service.train(input, output, limit, parameters);
    }

    /** Передаёт открытие поисковой сессии сервису Word2Vec. */
    @Override
    public SearchSession open(Path directory) throws IOException {
        return service.open(directory);
    }

    /** Передаёт открытие с метаданными сервису Word2Vec. */
    @Override
    public OpenedModel<Word2VecParameters> openModel(Path directory) {
        return service.openModel(directory);
    }
}
