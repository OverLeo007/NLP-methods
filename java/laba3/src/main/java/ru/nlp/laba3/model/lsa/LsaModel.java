package ru.nlp.laba3.model.lsa;

import java.util.List;

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
