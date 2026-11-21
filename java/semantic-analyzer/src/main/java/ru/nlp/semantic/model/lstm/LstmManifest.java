package ru.nlp.semantic.model.lstm;

import java.util.Map;

import lombok.Builder;

/** Версионированные метаданные LSTM и контрольные суммы полного комплекта. */
@Builder
public record LstmManifest(
        /** Идентификатор lstm. */
        String algorithm,
        /** Версия схемы кодировщика, словаря и ограничения длины. */
        int formatVersion,
        /** Архитектура и гиперпараметры запуска. */
        LstmParameters parameters,
        /** Профиль общей нормализации текста. */
        String normalization,
        /** SHA-256 встроенного стоп-списка. */
        String stopwordsSha256,
        /** Версии среды обучения. */
        Map<String, String> versions,
        /** Времена стадий общего сценария. */
        Map<String, Double> seconds,
        /** Число токенов корпуса до ограничения длины LSTM. */
        long tokens,
        /** Число документов с нулевым вектором. */
        long zeroDocuments,
        /** Полное число документов корпуса. */
        int documents,
        /** Число обычных терминов словаря. */
        int vocabularySize,
        /** Кросс-энтропия на токен фиксированного пакета до обучения. */
        double initialTokenLoss,
        /** Кросс-энтропия на токен того же пакета после обучения. */
        double finalTokenLoss,
        /** SHA-256 ZIP с конфигурацией сети и весами. */
        String weightsSha256,
        /** SHA-256 сжатого JSON корпуса и словаря. */
        String dataSha256
) {}
