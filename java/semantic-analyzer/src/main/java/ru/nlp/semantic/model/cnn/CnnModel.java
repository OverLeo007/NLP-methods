package ru.nlp.semantic.model.cnn;

import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;

import ru.nlp.semantic.model.cnn.dto.CnnParameters;
import ru.nlp.semantic.model.cnn.dto.CnnTrainingStatistics;

/** Обученное состояние CNN без файлов, корпуса и поискового индекса. */
public record CnnModel(
        /** Общий кодировщик документов и запросов; после обучения используется только для чтения. */
        MultiLayerNetwork network,
        /** Обычные термины; ID=позиция+2, PAD=0, OOV=1. */
        List<String> vocabulary,
        /** Архитектура и параметры запуска. */
        CnnParameters parameters,
        /** Сохранённая диагностика обучения. */
        CnnTrainingStatistics training
) {

    /** Фиксирует порядок словаря. */
    public CnnModel {
        vocabulary = List.copyOf(vocabulary);
    }
}
