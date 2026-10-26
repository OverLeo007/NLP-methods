package ru.nlp.laba3.model.word2vec;

import java.util.Map;

import ru.nlp.laba3.model.corpus.Corpus;

/** Данные для восстановления поиска; хранятся отдельно от метаданных в gzip JSON. */
public record Word2VecModelData(
        /** Документы и счётчики отбора корпуса. */
        Corpus corpus,
        /** Сглаженный IDF по токенам обучающей коллекции. */
        Map<String, Double> idf
) {}
