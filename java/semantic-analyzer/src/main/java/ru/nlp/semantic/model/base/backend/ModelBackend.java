package ru.nlp.semantic.model.base.backend;

import java.io.IOException;
import java.nio.file.Path;

import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.dto.OpenedModel;
import ru.nlp.semantic.model.base.dto.TrainingSummary;
import ru.nlp.semantic.search.SearchSession;

/** Точка входа фронтенда к сценариям выбранного анализатора с параметрами P. */
public interface ModelBackend<P> {

    /** Возвращает алгоритм реализации независимо от типа её параметров. */
    Algorithm algorithm();

    /** Обучает анализатор на исходном корпусе и сохраняет его в новый каталог. */
    TrainingSummary train(Path input, Path output, int limit, P parameters) throws IOException;

    /** Открывает сохранённую модель для повторных запросов без переобучения. */
    SearchSession open(Path directory) throws IOException;

    /** Открывает сессию вместе с типизированными параметрами и метаданными модели. */
    OpenedModel<P> openModel(Path directory) throws IOException;
}
