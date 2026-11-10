package ru.nlp.laba3.vectorization;

import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Обучение IDF по документам и построение разреженного sublinear TF-IDF. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class TfIdf {

    /** Вычисляет сглаженный IDF без участия оценочных запросов. */
    public static Map<String, Double> fitIdf(List<List<String>> documents) {
        var frequencies = new HashMap<String, Integer>();

        for (var tokens : documents) {
            new HashSet<>(tokens).forEach(token -> frequencies.merge(token, 1, Integer::sum));
        }

        var idf = new HashMap<String, Double>();
        frequencies.forEach(
                (token, frequency) -> idf.put(token, Math.log((1.0 + documents.size()) / (1.0 + frequency)) + 1.0)
        );

        return Map.copyOf(idf);
    }

    /** Вычисляет (1 + ln TF) × IDF и L2-норму только по известным терминам. */
    public static Map<Integer, Double> sublinearVector(
            List<String> tokens,
            Map<String, Integer> positions,
            double[] idf
    ) {
        var values = new HashMap<Integer, Double>();

        for (String token : tokens) {
            var position = positions.get(token);

            if (position != null) {
                values.merge(position, 1.0, Double::sum);
            }
        }

        values.replaceAll((position, count) -> (1 + Math.log(count)) * idf[position]);
        double norm = Math.sqrt(values.values().stream().mapToDouble(value -> value * value).sum());

        if (norm > 0) {
            values.replaceAll((position, value) -> value / norm);
        }

        return values;
    }
}
