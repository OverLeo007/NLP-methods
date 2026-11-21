package ru.nlp.semantic.model.lsa;

import java.util.List;

import ru.nlp.semantic.model.lsa.dto.LsaParameters;

/** Обученные преобразования LSA без документов, файлов и поискового индекса. */
public record LsaModel(
        /** Словарь в порядке строк проекции. */
        List<String> vocabulary,
        /** Сглаженный IDF терминов. */
        double[] idf,
        /** Термовые SVD-компоненты. */
        double[][] projection,
        /** Ведущие сингулярные числа. */
        double[] singularValues,
        /** Параметры запуска обучения. */
        LsaParameters parameters
) {}
