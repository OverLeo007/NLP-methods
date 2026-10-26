package ru.nlp.laba3.model.evaluation;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import lombok.Builder;

/** Состав и происхождение общей коллекции; запросы и разметка отделены от корпуса. */
@Builder
public record RiaSubsetManifest(
        /** Идентификатор правил отбора и трактовки отложенных запросов. */
        String protocol,
        /** Размеры и seed отбора. */
        SubsetParameters parameters,
        /** Профиль нормализации для проверки пригодности текстов. */
        String preprocessing,
        /** SHA-256 встроенного стоп-списка. */
        String stopwordsSha256,
        /** Версия Java, использованная при отборе. */
        String javaVersion,
        /** SHA-256 источников corpus, queries, qrels. */
        Map<String, String> sourceSha256,
        /** SHA-256 каждого выходного файла данных. */
        Map<String, String> outputSha256,
        /** ID запросов в порядке детерминированного отбора. */
        List<String> queryIds,
        /** ID статей в порядке записи в corpus.csv. */
        List<String> documentIds,
        /** Счётчики очистки и совпадений. */
        SubsetStatistics statistics
) {

    /** Фиксирует состав независимо от коллекций вызывающего кода. */
    public RiaSubsetManifest {
        sourceSha256 = Collections.unmodifiableMap(new TreeMap<>(sourceSha256));
        outputSha256 = Collections.unmodifiableMap(new TreeMap<>(outputSha256));
        queryIds = List.copyOf(queryIds);
        documentIds = List.copyOf(documentIds);
    }
}
