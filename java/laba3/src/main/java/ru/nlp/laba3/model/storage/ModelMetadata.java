package ru.nlp.laba3.model.storage;

import java.util.Map;

/** Общие метаданные модели с типизированными параметрами алгоритма. */
public record ModelMetadata<P>(
        /** Параметры, сохранённые при обучении. */
        P parameters,
        /** Профиль нормализации. */
        String normalization,
        /** SHA-256 стоп-списка. */
        String stopwordsSha256,
        /** Версии библиотек и среда обучения. */
        Map<String, String> versions,
        /** Сохранённые времена стадий обучения и записи. */
        Map<String, Double> seconds,
        /** Число обучающих токенов; null при отсутствии старого счётчика. */
        Long tokens
) {

    /** Защищает снимки метаданных от изменений вызывающего кода. */
    public ModelMetadata {
        versions = Map.copyOf(versions);
        seconds = Map.copyOf(seconds);
    }
}
