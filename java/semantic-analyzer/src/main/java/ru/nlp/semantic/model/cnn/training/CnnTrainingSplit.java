package ru.nlp.semantic.model.cnn.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Random;

import ru.nlp.semantic.exception.InvalidInputException;

/** Разделение нормализованных документов до построения словаря, без запросов и qrels. */
public record CnnTrainingSplit(
        /** Документы для словаря и обновления весов. */
        List<List<String>> training,
        /** Отложенные документы только для контрастивной validation-потери. */
        List<List<String>> validation
) {

    /** Фиксирует списки, включая вложенные последовательности токенов. */
    public CnnTrainingSplit {
        training = training.stream().map(List::copyOf).toList();
        validation = validation.stream().map(List::copyOf).toList();
    }

    /** Делит группы одинаковых токенов целиком, сохраняя их частоты внутри части. */
    public static CnnTrainingSplit create(List<List<String>> documents, double fraction, long seed) {
        if (!Double.isFinite(fraction) || fraction < 0 || fraction > 0.5) {
            throw new InvalidInputException("CNN: validation-fraction должна быть от 0 до 0.5");
        }

        if (fraction == 0) {
            return new CnnTrainingSplit(documents, List.of());
        }
        var grouped = new LinkedHashMap<List<String>, List<List<String>>>();

        for (var document : documents) {
            if (!document.isEmpty()) {
                grouped.computeIfAbsent(List.copyOf(document), ignored -> new ArrayList<>()).add(document);
            }
        }

        var groups = new ArrayList<>(grouped.values());
        int validationGroups = Math.max(2, (int) Math.round(groups.size() * fraction));

        if (groups.size() - validationGroups < 2) {
            throw new InvalidInputException(
                    "CNN: для validation нужны минимум четыре различные последовательности токенов"
            );
        }
        Collections.shuffle(groups, new Random(seed));
        var validation = groups.subList(0, validationGroups).stream().flatMap(List::stream).toList();
        var training = groups.subList(validationGroups, groups.size()).stream().flatMap(List::stream).toList();

        return new CnnTrainingSplit(training, validation);
    }
}
