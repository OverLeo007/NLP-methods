package ru.nlp.laba3.preprocessing;

import java.util.List;
import java.util.stream.Collectors;

import lombok.Builder;
import lombok.Singular;

/** Неизменяемая цепочка правил с явным порядком применения. */
public final class TextNormalizationChain {

    private final List<TextNormalizationRule> rules;

    /** Фиксирует порядок правил независимо от изменений исходного списка. */
    @Builder
    private TextNormalizationChain(@Singular List<TextNormalizationRule> rules) {
        this.rules = List.copyOf(rules);
    }

    /** Применяет каждое правило к результату предыдущего. */
    public String normalize(String text) {
        String normalized = text;

        for (TextNormalizationRule rule : rules) {
            normalized = rule.normalize(normalized);
        }

        return normalized;
    }

    /** Возвращает идентификаторы правил с учётом их порядка. */
    public String profile() {
        return rules.stream().map(TextNormalizationRule::id).collect(Collectors.joining(" -> "));
    }
}
