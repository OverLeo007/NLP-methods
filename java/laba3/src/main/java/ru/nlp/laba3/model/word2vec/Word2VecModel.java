package ru.nlp.laba3.model.word2vec;

import java.util.Map;

import org.deeplearning4j.models.word2vec.Word2Vec;

/** Состояние Word2Vec для сохранения и построения векторизатора. */
public record Word2VecModel(
        /** Обученные embeddings; доступ к нативным данным требует NativeRuntime. */
        Word2Vec vectors,
        /** IDF, обученный по документам коллекции. */
        Map<String, Double> idf,
        /** Параметры запуска обучения. */
        Word2VecParameters parameters,
        /** Зафиксированные настройки алгоритма и агрегации. */
        Map<String, String> trainingSettings
) {

    /** Фиксирует словарь IDF и настройки независимо от исходных отображений. */
    public Word2VecModel {
        idf = Map.copyOf(idf);
        trainingSettings = Map.copyOf(trainingSettings);
    }
}
