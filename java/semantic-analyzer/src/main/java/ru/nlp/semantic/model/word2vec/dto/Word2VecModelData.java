package ru.nlp.semantic.model.word2vec.dto;

import java.util.Map;

import ru.nlp.semantic.corpus.dto.Corpus;

/** Данные для восстановления поиска; хранятся отдельно от метаданных в gzip JSON. */
public record Word2VecModelData(
        /** Документы и счётчики отбора корпуса. */
        Corpus corpus,
        /** Сглаженный IDF по токенам обучающей коллекции. */
        Map<String, Double> idf
) {}
