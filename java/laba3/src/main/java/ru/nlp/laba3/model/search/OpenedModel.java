package ru.nlp.laba3.model.search;

import ru.nlp.laba3.model.storage.ModelMetadata;
import ru.nlp.laba3.search.SearchSession;

/** Поисковая сессия вместе с происхождением загруженного состояния. */
public record OpenedModel<P>(
        /** Готовая сессия без повторного обучения. */
        SearchSession session,
        /** Параметры и общие метаданные модели. */
        ModelMetadata<P> metadata
) {}
