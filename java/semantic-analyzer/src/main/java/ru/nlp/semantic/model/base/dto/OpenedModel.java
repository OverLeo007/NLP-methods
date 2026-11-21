package ru.nlp.semantic.model.base.dto;

import ru.nlp.semantic.search.SearchSession;

/** Поисковая сессия вместе с происхождением загруженного состояния. */
public record OpenedModel<P>(
        /** Готовая сессия без повторного обучения. */
        SearchSession session,
        /** Параметры и общие метаданные модели. */
        ModelMetadata<P> metadata
) {}
