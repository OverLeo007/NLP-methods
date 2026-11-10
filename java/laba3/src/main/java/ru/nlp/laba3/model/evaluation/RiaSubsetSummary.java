package ru.nlp.laba3.model.evaluation;

/** Итог команды подготовки; изменяющееся время не входит в манифест состава. */
public record RiaSubsetSummary(
        /** Каталог подготовленной коллекции. */
        String output,
        /** Счётчики отбора. */
        SubsetStatistics statistics,
        /** Полное время подготовки в секундах. */
        double seconds
) {}
