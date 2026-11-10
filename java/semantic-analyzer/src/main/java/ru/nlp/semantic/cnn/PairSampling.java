package ru.nlp.semantic.cnn;

import java.util.Arrays;
import java.util.Random;

/** Способ выбора двух положительных фрагментов; архитектура кодировщика от него не зависит. */
public enum PairSampling {

    /** Два независимых фрагмента длиной от половины доступного предела до предела. */
    symmetric {

        @Override
        public Fragments sample(int[] document, int maxLength, Random random) {
            int maximum = Math.min(maxLength, document.length);
            int minimum = Math.max(1, maximum / 2);

            return new Fragments(span(document, minimum, maximum, random), span(document, minimum, maximum, random));
        }
    },
    /** Независимый короткий фрагмент 4–16 токенов и длинный по прежнему правилу. */
    short_long {

        @Override
        public Fragments sample(int[] document, int maxLength, Random random) {
            int maximum = Math.min(maxLength, document.length);
            int shortMaximum = Math.min(16, maximum);

            return new Fragments(
                    span(document, Math.min(4, shortMaximum), shortMaximum, random),
                    span(document, Math.max(1, maximum / 2), maximum, random)
            );
        }
    };

    /** Выбирает два фрагмента документа с известными токенами по переданному генератору. */
    public abstract Fragments sample(int[] document, int maxLength, Random random);

    private static int[] span(int[] document, int minimum, int maximum, Random random) {
        for (int attempt = 0; attempt < 16; attempt++) {
            int length = random.nextInt(maximum - minimum + 1) + minimum;
            int start = random.nextInt(document.length - length + 1);
            int[] fragment = Arrays.copyOfRange(document, start, start + length);

            if (CnnVocabulary.knownTokens(fragment) > 0) {
                return fragment;
            }
        }

        int known = 0;

        while (document[known] < 2) {
            known++;
        }

        int start = Math.min(known, document.length - maximum);

        return Arrays.copyOfRange(document, start, start + maximum);
    }

    /** Две строки одной положительной пары, без меток поисковой релевантности. */
    public record Fragments(
            /** Первая строка; в short_long она короткая. */
            int[] first,
            /** Вторая строка; в short_long она длинная. */
            int[] second
    ) {}
}
