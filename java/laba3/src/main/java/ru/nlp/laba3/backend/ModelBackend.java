package ru.nlp.laba3.backend;

import java.io.IOException;
import java.nio.file.Path;

import ru.nlp.laba3.model.search.OpenedModel;
import ru.nlp.laba3.model.training.TrainingSummary;
import ru.nlp.laba3.search.SearchSession;

/** Точка входа фронтенда к сценариям выбранного анализатора с параметрами P. */
public interface ModelBackend<P> {

    /** Обучает анализатор на исходном корпусе и сохраняет его в новый каталог. */
    TrainingSummary train(Path input, Path output, int limit, P parameters) throws IOException;

    /** Открывает сохранённую модель для повторных запросов без переобучения. */
    SearchSession open(Path directory) throws IOException;

    /** Открывает сессию вместе с типизированными параметрами и метаданными модели. */
    OpenedModel<P> openModel(Path directory) throws IOException;
}
