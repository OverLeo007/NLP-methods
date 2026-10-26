package ru.nlp.laba3.model.search;

import java.util.List;

/** Статус текстового запроса, покрытие и неизменяемая ранжированная выдача. */
public record SearchResult(
        /** Статус обработки запроса. */
        String status,
        /** Число известных вхождений токенов запроса. */
        int knownTokens,
        /** Общее число токенов запроса. */
        int totalTokens,
        /** Ранжированные результаты поиска. */
        List<SearchHit> hits
) {

    /** Защищает опубликованную выдачу от последующего изменения списка. */
    public SearchResult {
        hits = List.copyOf(hits);
    }
}
