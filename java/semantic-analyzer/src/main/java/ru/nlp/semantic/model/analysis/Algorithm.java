package ru.nlp.semantic.model.analysis;

import lombok.RequiredArgsConstructor;

/** Идентификаторы CLI и единые отображаемые имена поддерживаемых анализаторов. */
@RequiredArgsConstructor
public enum Algorithm {

    /** TF-IDF-взвешенное среднее embeddings Word2Vec. */
    word2vec("Word2Vec"),
    /** TF-IDF с усечённым SVD. */
    lsa("LSA"),
    /** Контрастивно обученный свёрточный кодировщик текста. */
    cnn("CNN"),
    /** LSTM-кодировщик, обученный восстанавливать последовательность токенов. */
    lstm("LSTM");

    private final String displayName;

    /** Возвращает имя алгоритма для логов и диагностических сообщений. */
    public String displayName() {
        return displayName;
    }
}
