package ru.nlp.laba3.vectorization;

import java.util.List;
import java.util.Map;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;

import ru.nlp.laba3.runtime.NativeRuntime;

/** TF-IDF-взвешенное среднее embeddings с линейным TF и защитой нативного чтения. */
public final class Word2VecVectorizer implements TextVectorizer {

    private final WordVectors model;

    private final Map<String, Double> idf;

    private final int dimensions;

    /** Связывает обученные embeddings с неизменяемым IDF документов. */
    public Word2VecVectorizer(WordVectors model, Map<String, Double> idf) {
        this.model = model;
        this.idf = Map.copyOf(idf);
        dimensions = NativeRuntime.withLock(() -> model.lookupTable().layerSize());
    }

    /** Возвращает размерность embeddings. */
    @Override
    public int dimensions() {
        return dimensions;
    }

    /** Читает embeddings под общим монитором нативной среды. */
    @Override
    public EncodedText encode(List<String> tokens) {
        return NativeRuntime.withLock(() -> vector(tokens));
    }

    private EncodedText vector(List<String> tokens) {
        double[] values = new double[dimensions];
        int known = 0;
        // Повторение токена реализует линейный TF; IDF обучен только на документах.

        for (String token : tokens) {
            if (!idf.containsKey(token) || !model.hasWord(token)) {
                continue;
            }
            var word = model.getWordVector(token);
            known++;

            for (int i = 0; i < dimensions; i++) {
                values[i] += idf.get(token) * word[i];
            }
        }

        double norm = 0;

        for (double value : values) {
            norm += value * value;
        }

        norm = Math.sqrt(norm);

        if (norm > 0 && Double.isFinite(norm)) {
            for (int i = 0; i < dimensions; i++) {
                values[i] /= norm;
            }
        } else {
            java.util.Arrays.fill(values, 0);
        }

        return new EncodedText(values, known);
    }
}
