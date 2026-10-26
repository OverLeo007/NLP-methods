package ru.nlp.laba3.runtime;

import java.util.LinkedHashMap;
import java.util.Map;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Общие монотонные замеры этапов одного запуска, без разделяемого состояния. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class StageTimings {

    /** Возвращает прошедшие секунды по метке System.nanoTime. */
    public static double elapsed(long started) {
        return (System.nanoTime() - started) / 1e9;
    }

    /** Добавляет время записи данных и контрольных сумм до записи самого манифеста. */
    public static Map<String, Double> withSaving(Map<String, Double> previous, long started) {
        var timings = new LinkedHashMap<>(previous);
        timings.put("saving", elapsed(started));

        return Map.copyOf(timings);
    }
}
