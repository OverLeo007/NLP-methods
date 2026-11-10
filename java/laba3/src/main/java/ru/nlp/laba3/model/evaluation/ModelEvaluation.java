package ru.nlp.laba3.model.evaluation;

import java.util.List;
import java.util.Map;

import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.storage.ModelMetadata;

/** Единый результат для любого алгоритма с типизированными параметрами обучения. */
public record ModelEvaluation<P>(
        /** Алгоритм анализатора. */
        Algorithm algorithm,
        /** Путь исходного комплекта модели. */
        String model,
        /** Контрольные суммы файлов комплекта. */
        Map<String, String> modelSha256,
        /** Параметры, версии и времена из сохранённой модели. */
        ModelMetadata<P> metadata,
        /** Число всех документов, включая нулевые. */
        int documents,
        /** Число документов без поискового вектора. */
        long zeroDocuments,
        /** ID всех документов без поискового вектора. */
        List<String> zeroDocumentIds,
        /** Время openModel, включая чтение, проверку и создание индекса. */
        double openingSeconds,
        /** Итог по полному набору запросов. */
        RetrievalMetrics metrics,
        /** Все запросы с выдачей и индивидуальными вкладами. */
        List<QueryEvaluation> results
) {

    /** Защищает результаты и контрольные суммы. */
    public ModelEvaluation {
        modelSha256 = Map.copyOf(modelSha256);
        zeroDocumentIds = List.copyOf(zeroDocumentIds);
        results = List.copyOf(results);
    }
}
