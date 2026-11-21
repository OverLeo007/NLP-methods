package ru.nlp.semantic.generation;

import ru.nlp.semantic.search.dto.SearchHit;
import ru.nlp.semantic.search.dto.SearchResult;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Русскоязычные шаблоны ответа по выдаче семантического поиска. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class TemplateGenerator {

    /** Подставляет в шаблон запрос, источник и цитату из первого результата. */
    public static String generate(String query, SearchResult result) {
        if (result.hits().isEmpty()) {
            return "По запросу «%s» документы не найдены.".formatted(query);
        }
        var hit = result.hits().getFirst();
        String source = hit.title() == null || hit.title().isBlank()
                ? "«%s»".formatted(hit.id())
                : "«%s» (ID %s)".formatted(hit.title(), hit.id());

        return "По запросу «%s» первым найден документ %s. В нём сказано: %s".formatted(query, source, quote(hit));
    }

    private static String quote(SearchHit hit) {
        String excerpt = hit.excerpt().strip();

        if (hit.excerptTruncated()) {
            return "«" + excerpt + "…»";
        }

        if (excerpt.endsWith("!") || excerpt.endsWith("?") || excerpt.endsWith("…") || excerpt.endsWith("...")) {
            return "«" + excerpt + "»";
        }

        if (excerpt.endsWith(".")) {
            excerpt = excerpt.substring(0, excerpt.length() - 1);
        }

        return "«" + excerpt + "».";
    }
}
