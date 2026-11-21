package ru.nlp.semantic.generation;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;

import org.junit.jupiter.api.Test;

import ru.nlp.semantic.config.JsonConfiguration;
import ru.nlp.semantic.model.corpus.Document;
import ru.nlp.semantic.model.search.SearchResult;
import ru.nlp.semantic.search.CosineIndex;

class TemplateGeneratorTest {

    @Test
    void marksOnlyTruncatedQuotesAndPreservesSearchJson() {
        String prefix = "я".repeat(179);
        var documents = List.of(
                new Document("a", "Заголовок", prefix + "!"),
                new Document("b", "", prefix + "🚀 дальше")
        );
        var index = new CosineIndex(documents, List.of(new double[]{1, 0}, new double[]{0, 1}), 2);

        var complete = new SearchResult("ok", 1, 1, index.search(new double[]{1, 0}, 2));
        assertThat(TemplateGenerator.generate("текст", complete))
                .contains("«Заголовок» (ID a)").endsWith("«" + prefix + "!»");

        var truncated = new SearchResult("ok", 1, 1, index.search(new double[]{0, 1}, 2));
        assertThat(TemplateGenerator.generate("текст", truncated))
                .contains("документ «b»").endsWith("«" + prefix + "…»");
        var json = new JsonConfiguration().jsonMapper().valueToTree(truncated.hits().getFirst());
        assertThat(json.propertyStream().map(entry -> entry.getKey()))
                .containsExactlyInAnyOrder("id", "title", "cosine", "excerpt");
    }
}
