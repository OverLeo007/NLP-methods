package ru.nlp.semantic.evaluation.dto;

/** Итог команды подготовки; изменяющееся время не входит в манифест состава. */
public record RiaSubsetSummary(
        /** Каталог подготовленной коллекции. */
        String output,
        /** Счётчики отбора. */
        SubsetStatistics statistics,
        /** Полное время подготовки в секундах. */
        double seconds
) {}
