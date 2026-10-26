package ru.nlp.laba3.model.storage;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import ru.nlp.laba3.model.corpus.Corpus;

/** Типизированный комплект: состояние алгоритма, корпус и общие метаданные. */
public record ModelBundle<M>(
        /** Отобранный корпус и счётчики его чтения. */
        Corpus corpus,
        /** Состояние конкретного алгоритма. */
        M model,
        /** Профиль нормализации. */
        String normalization,
        /** SHA-256 встроенного стоп-списка. */
        String stopwordsSha256,
        /** Версии библиотек и сведения о платформе. */
        Map<String, String> versions,
        /** Число токенов, либо null, если загруженный комплект не содержит счётчик. */
        Long tokens,
        /** Число нулевых документов, либо null, если счётчик не сохранён. */
        Long zeroDocuments,
        /** Времена этапов в секундах. */
        Map<String, Double> seconds
) {

    /** Защищает общие метаданные и сохраняет порядок сведений об окружении. */
    public ModelBundle {
        versions = Collections.unmodifiableMap(new LinkedHashMap<>(versions));
        seconds = Map.copyOf(seconds);
    }
}
