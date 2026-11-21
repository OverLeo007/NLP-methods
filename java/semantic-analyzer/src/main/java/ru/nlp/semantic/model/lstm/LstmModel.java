package ru.nlp.semantic.model.lstm;

import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;

/** Обученный граф, словарь и ошибка восстановления на фиксированном пакете. */
public record LstmModel(
        /** Кодировщик, декодер и выходной слой с обученными весами. */
        ComputationGraph network,
        /** Термины в порядке ID начиная с 2. */
        List<String> vocabulary,
        /** Параметры обучения. */
        LstmParameters parameters,
        /** Средняя кросс-энтропия на известный токен до обучения. */
        double initialTokenLoss,
        /** Та же оценка на том же пакете после обучения. */
        double finalTokenLoss
) {

    /** Фиксирует порядок словаря. */
    public LstmModel {
        vocabulary = List.copyOf(vocabulary);
    }
}
