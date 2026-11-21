package ru.nlp.semantic.model.lstm;

import java.util.List;

import org.deeplearning4j.nn.graph.ComputationGraph;

import ru.nlp.semantic.model.lstm.dto.LstmParameters;

/** Обученный граф, словарь и потеря выбранной задачи на фиксированном пакете. */
public record LstmModel(
        /** Кодировщик и слои выбранной задачи обучения. */
        ComputationGraph network,
        /** Термины в порядке ID начиная с 2. */
        List<String> vocabulary,
        /** Параметры обучения. */
        LstmParameters parameters,
        /** Значение потери до обучения кодировщика. */
        double initialLoss,
        /** Та же оценка на том же пакете после обучения. */
        double finalLoss
) {

    /** Фиксирует порядок словаря. */
    public LstmModel {
        vocabulary = List.copyOf(vocabulary);
    }
}
