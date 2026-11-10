package ru.nlp.laba3.vectorization;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/** Преобразование токенов через обученные IDF и SVD-проекцию без нативных операций. */
public final class LsaVectorizer implements TextVectorizer {

    private final Map<String, Integer> positions;

    private final double[] idf;

    private final double[][] projection;

    private final int dimensions;

    /** Создаёт независимый снимок словаря, IDF и проекции обученной модели. */
    public LsaVectorizer(List<String> vocabulary, double[] idf, double[][] projection) {
        var fittedPositions = new HashMap<String, Integer>();

        for (int i = 0; i < vocabulary.size(); i++) {
            fittedPositions.put(vocabulary.get(i), i);
        }

        positions = Map.copyOf(fittedPositions);
        this.idf = idf.clone();
        this.projection = Arrays.stream(projection).map(double[]::clone).toArray(double[][]::new);
        dimensions = projection[0].length;
    }

    /** Возвращает число сохранённых SVD-компонент. */
    @Override
    public int dimensions() {
        return dimensions;
    }

    /** Вычисляет sublinear TF-IDF × V_k, покрытие и L2-нормализацию проекции. */
    @Override
    public EncodedText encode(List<String> tokens) {
        int known = (int) tokens.stream().filter(positions::containsKey).count();
        double[] vector = new double[dimensions];
        TfIdf.sublinearVector(tokens, positions, idf).forEach((position, weight) -> {
            for (int j = 0; j < vector.length; j++) {
                vector[j] += weight * projection[position][j];
            }
        });
        double norm = Math.sqrt(Arrays.stream(vector).map(value -> value * value).sum());
        // Численный шум усечённого подпространства не превращается в единичный вектор.

        if (norm > 1e-12 && Double.isFinite(norm)) {
            for (int j = 0; j < vector.length; j++) {
                vector[j] /= norm;
            }
        } else {
            Arrays.fill(vector, 0);
        }

        return new EncodedText(vector, known);
    }
}
