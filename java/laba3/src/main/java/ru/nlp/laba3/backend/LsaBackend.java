package ru.nlp.laba3.backend;

import java.io.IOException;
import java.nio.file.Path;

import org.springframework.stereotype.Component;

import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.model.search.OpenedModel;
import ru.nlp.laba3.model.training.TrainingSummary;
import ru.nlp.laba3.search.SearchSession;
import ru.nlp.laba3.service.LsaService;

import lombok.RequiredArgsConstructor;

/** Адаптер фронтенда к сценариям сервиса LSA. */
@Component
@RequiredArgsConstructor
public class LsaBackend implements ModelBackend<LsaParameters> {

    private final LsaService service;

    /** Передаёт обучение и сохранение сервису LSA. */
    @Override
    public TrainingSummary train(Path input, Path output, int limit, LsaParameters parameters) throws IOException {
        return service.train(input, output, limit, parameters);
    }

    /** Передаёт открытие поисковой сессии сервису LSA. */
    @Override
    public SearchSession open(Path directory) throws IOException {
        return service.open(directory);
    }

    /** Передаёт открытие с метаданными сервису LSA. */
    @Override
    public OpenedModel<LsaParameters> openModel(Path directory) {
        return service.openModel(directory);
    }
}
