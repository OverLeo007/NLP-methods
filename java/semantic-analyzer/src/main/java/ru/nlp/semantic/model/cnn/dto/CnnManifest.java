package ru.nlp.semantic.model.cnn.dto;

import java.util.Map;

import lombok.Builder;

/** Версионированные метаданные CNN и контрольные суммы полного комплекта. */
@Builder
public record CnnManifest(
        /** Идентификатор cnn. */
        String algorithm,
        /** Версия схемы кодировщика, словаря и ограничения длины. */
        int formatVersion,
        /** Архитектура и гиперпараметры запуска. */
        CnnParameters parameters,
        /** Профиль общей нормализации текста. */
        String normalization,
        /** SHA-256 встроенного стоп-списка. */
        String stopwordsSha256,
        /** Версии среды обучения. */
        Map<String, String> versions,
        /** Времена стадий общего сценария. */
        Map<String, Double> seconds,
        /** Число токенов корпуса до ограничения длины CNN. */
        long tokens,
        /** Число документов с нулевым вектором. */
        long zeroDocuments,
        /** Полное число документов корпуса. */
        int documents,
        /** Число обычных терминов словаря. */
        int vocabularySize,
        /** Диагностика выполненного обучения. */
        CnnTrainingStatistics training,
        /** SHA-256 ZIP с конфигурацией сети и весами. */
        String weightsSha256,
        /** SHA-256 сжатого JSON корпуса и словаря. */
        String dataSha256
) {}
