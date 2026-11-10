package ru.nlp.semantic.model.cnn;

import java.util.List;

import ru.nlp.semantic.model.corpus.Corpus;

/** Корпус и словарь CNN; веса записываются отдельно средствами DL4J. */
public record CnnModelData(
        /** Полный корпус в исходном порядке, включая документы с нулевыми векторами. */
        Corpus corpus,
        /** Термины в порядке ID без PAD/OOV. */
        List<String> vocabulary
) {}
