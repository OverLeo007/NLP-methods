package ru.nlp.semantic.config;

import static ru.nlp.semantic.preprocessing.rules.NormalizationRuleFactory.create;
import static ru.nlp.semantic.preprocessing.rules.NormalizationRuleType.LOWERCASE;
import static ru.nlp.semantic.preprocessing.rules.NormalizationRuleType.NFC;
import static ru.nlp.semantic.preprocessing.rules.NormalizationRuleType.RUSSIAN_YO;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import ru.nlp.semantic.preprocessing.TextNormalizationChain;

/** Явная композиция правил нормализации документов и запросов. */
@Configuration(proxyBeanMethods = false)
public class PreprocessingConfiguration {

    /** Создаёт неизменяемую цепочку NFC → lowercase → ё/е. */
    @Bean
    public TextNormalizationChain textNormalizationChain() {
        return TextNormalizationChain.builder()
                .rule(create(NFC))
                .rule(create(LOWERCASE))
                .rule(create(RUSSIAN_YO))
                .build();
    }
}
