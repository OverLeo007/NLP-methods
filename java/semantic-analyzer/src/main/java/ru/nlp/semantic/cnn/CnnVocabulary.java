package ru.nlp.semantic.cnn;

import java.util.List;

import ru.nlp.semantic.vectorization.SequenceVocabulary;

/** Словарь CNN с общими для нейросетевых моделей ID токенов и масками. */
public final class CnnVocabulary extends SequenceVocabulary {

    /** Восстанавливает исходный порядок ID модели CNN. */
    public CnnVocabulary(List<String> words) {
        super(words);
    }

    /** Отбирает термины обучающих документов. */
    public static CnnVocabulary fit(List<List<String>> documents, int minCount, int maxFeatures) {
        return new CnnVocabulary(selectWords(documents, minCount, maxFeatures));
    }
}
