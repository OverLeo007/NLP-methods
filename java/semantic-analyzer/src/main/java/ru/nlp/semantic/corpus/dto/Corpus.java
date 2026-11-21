package ru.nlp.semantic.corpus.dto;

import java.util.List;

/** Выборка и счётчики только просмотренной части источника. */
public record Corpus(
        /** Отобранные документы корпуса. */
        List<Document> documents,
        /** Число просмотренных записей. */
        long scanned,
        /** Число пустых записей. */
        long empty,
        /** Число дубликатов. */
        long duplicates
) {}
