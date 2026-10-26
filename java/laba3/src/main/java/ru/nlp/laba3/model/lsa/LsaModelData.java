package ru.nlp.laba3.model.lsa;

import java.util.List;
import java.util.Map;

import ru.nlp.laba3.model.corpus.Corpus;

/** Достаточные данные для преобразования запросов и восстановления LSA-индекса. */
public record LsaModelData(
        /** Отобранные документы и счётчики чтения. */
        Corpus corpus,
        /** Упорядоченный словарь, обученный только на документах. */
        List<String> vocabulary,
        /** Сглаженный IDF в порядке словаря. */
        double[] idf,
        /** Правые сингулярные векторы TF-IDF: термин × компонента. */
        double[][] projection,
        /** Сохранённые ведущие сингулярные числа. */
        double[] singularValues,
        /** Параметры запуска. */
        LsaParameters parameters,
        /** Профиль нормализации. */
        String normalization,
        /** Контрольная сумма стоп-списка. */
        String stopwordsSha256,
        /** Версии библиотек и сведения об окружении. */
        Map<String, String> versions,
        /** Замеры завершённых стадий до сохранения; полный набор находится в манифесте. */
        Map<String, Double> seconds,
        /** Число токенов; может отсутствовать в ранее сохранённом комплекте. */
        Long tokens,
        /** Число нулевых документов; может отсутствовать в ранее сохранённом комплекте. */
        Long zeroDocuments
) {}
