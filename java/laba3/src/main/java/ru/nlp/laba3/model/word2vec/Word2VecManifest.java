package ru.nlp.laba3.model.word2vec;

import java.util.Map;

import lombok.Builder;

/** Метаданные фиксированной структуры без документов и словарей корпуса. */
@Builder
public record Word2VecManifest(
        /** Параметры обучения Word2Vec. */
        Word2VecParameters parameters,
        /** Профиль и порядок нормализации. */
        String normalization,
        /** SHA-256 встроенного стоп-списка. */
        String stopwordsSha256,
        /** Способ выбора документов. */
        String selection,
        /** Число отобранных документов. */
        int documents,
        /** Число просмотренных записей источника. */
        long scanned,
        /** Число исключённых пустых записей. */
        long empty,
        /** Число исключённых дубликатов. */
        long duplicates,
        /** Версии библиотек и параметры платформы. */
        Map<String, String> versions,
        /** SHA-256 файла embeddings. */
        String weightsSha256,
        /** SHA-256 сжатого файла документов и IDF. */
        String dataSha256,
        /** Число нормализованных токенов. */
        long tokens,
        /** Число слов в embeddings. */
        int vocabularySize,
        /** Число документов с нулевым вектором. */
        long zeroDocuments,
        /** Фиксированные настройки алгоритма. */
        Map<String, String> trainingSettings,
        /** Длительности этапов в секундах. */
        Map<String, Double> seconds
) {}
