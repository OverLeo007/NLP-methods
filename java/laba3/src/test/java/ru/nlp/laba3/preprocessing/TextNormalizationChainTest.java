package ru.nlp.laba3.preprocessing;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.preprocessing.rules.LowercaseRule;
import ru.nlp.laba3.preprocessing.rules.NormalizationRuleFactory;
import ru.nlp.laba3.preprocessing.rules.RussianYoRule;
import ru.nlp.laba3.preprocessing.rules.UnicodeNfcRule;

class TextNormalizationChainTest {

    @Test
    void ruleOrderAndProfileRemainFixedWhenSourceListChanges() {
        var rules = new ArrayList<>(
                List.of(
                        new UnicodeNfcRule(),
                        new LowercaseRule(),
                        new RussianYoRule()
                )
        );
        var builder = TextNormalizationChain.builder()
                .rules(rules);
        var chain = builder.build();
        builder.rule(new LowercaseRule());
        assertThat(builder.build().profile()).isEqualTo("nfc -> lower -> yo -> lower");
        rules.clear();

        assertThat(chain.normalize("Е\u0308ЛКА")).isEqualTo("елка");
        assertThat(chain.profile()).isEqualTo("nfc -> lower -> yo");

        var reordered = TextNormalizationChain.builder()
                .rule(new RussianYoRule())
                .rule(new LowercaseRule())
                .build();
        assertThat(reordered.normalize("ЁЛКА")).isEqualTo("ёлка");
        assertThat(reordered.profile()).isEqualTo("yo -> lower");
    }

    @Test
    void stringRuleFactoryPreservesIdentifiersAndRejectsUnknownOnes() {
        assertThat(NormalizationRuleFactory.create("nfc").id()).isEqualTo("nfc");
        assertThat(NormalizationRuleFactory.create("lower").normalize("ЁЛКА"))
                .isEqualTo("ёлка");
        assertThat(NormalizationRuleFactory.create("yo").normalize("ёлка"))
                .isEqualTo("елка");
        assertThatThrownBy(() -> NormalizationRuleFactory.create("missing"))
                .isInstanceOf(InvalidInputException.class);
        assertThatThrownBy(() -> NormalizationRuleFactory.create(null))
                .isInstanceOf(InvalidInputException.class);
    }
}
