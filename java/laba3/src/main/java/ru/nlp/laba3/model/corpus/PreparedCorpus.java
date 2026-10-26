package ru.nlp.laba3.model.corpus;

import java.util.List;

/** Общий неизменяемый результат чтения и нормализации обучающей коллекции. */
public record PreparedCorpus(
        /** Отобранные документы с исходными ID, текстами и счётчиками отбора. */
        Corpus corpus,
        /** Нормализованные токены в том же порядке, что документы корпуса. */
        List<List<String>> documents,
        /** Общее число нормализованных токенов. */
        long tokens
) {

    /** Защищает порядок документов и вложенные списки токенов от изменения. */
    public PreparedCorpus {
        corpus = new Corpus(List.copyOf(corpus.documents()), corpus.scanned(), corpus.empty(), corpus.duplicates());
        documents = documents.stream().map(List::copyOf).toList();
    }
}
