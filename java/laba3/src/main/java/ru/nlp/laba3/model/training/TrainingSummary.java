package ru.nlp.laba3.model.training;

import java.util.Map;

/** Общий итог обучения анализатора, пригодный для вывода любым фронтендом. */
public record TrainingSummary(
        /** Число отобранных документов корпуса. */
        int documents,
        /** Число нормализованных токенов. */
        long tokens,
        /** Число терминов обученного словаря. */
        int vocabularySize,
        /** Фактическая размерность векторов. */
        int dimensions,
        /** Число документов с нулевым вектором. */
        long zeroDocuments,
        /** Полное время вызова тренера, включая ожидание нативного монитора и инициализацию. */
        double trainingSeconds,
        /** Каталог сохранённой модели. */
        String output,
        /** Времена стадий preparation, training, indexing, saving в секундах. */
        Map<String, Double> seconds
) {

    /** Фиксирует завершённые замеры независимо от исходной карты. */
    public TrainingSummary {
        seconds = Map.copyOf(seconds);
    }
}
