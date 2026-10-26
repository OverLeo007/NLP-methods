package ru.nlp.laba3.model.analysis;

import lombok.RequiredArgsConstructor;

/** Идентификаторы CLI и единые отображаемые имена поддерживаемых анализаторов. */
@RequiredArgsConstructor
public enum Algorithm {

    /** TF-IDF-взвешенное среднее embeddings Word2Vec. */
    word2vec("Word2Vec"),
    /** TF-IDF с усечённым SVD. */
    lsa("LSA");

    private final String displayName;

    /** Возвращает имя алгоритма для логов и диагностических сообщений. */
    public String displayName() {
        return displayName;
    }
}
