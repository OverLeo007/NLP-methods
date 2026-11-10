package ru.nlp.semantic.vectorization;

import java.util.Arrays;
import java.util.List;

import ru.nlp.semantic.cnn.CnnVocabulary;
import ru.nlp.semantic.model.cnn.CnnModel;
import ru.nlp.semantic.runtime.NativeRuntime;

/** CNN-векторизация ограниченного префикса текста с явными PAD/OOV и L2-нормализацией. */
public final class CnnVectorizer implements TextVectorizer {

    private final CnnModel model;

    private final CnnVocabulary vocabulary;

    /** Связывает обученный кодировщик с его словарём и правилами длины. */
    public CnnVectorizer(CnnModel model) {
        this.model = model;
        vocabulary = new CnnVocabulary(model.vocabulary());
    }

    /** Возвращает размер представления текста. */
    @Override
    public int dimensions() {
        return model.parameters().dimensions();
    }

    /** Кодирует первые maxLength токенов; knownTokens считает только использованные вхождения. */
    @Override
    public EncodedText encode(List<String> tokens) {
        int[] ids = vocabulary.encode(tokens.subList(0, Math.min(tokens.size(), model.parameters().maxLength())));
        int known = CnnVocabulary.knownTokens(ids);

        if (known == 0) {
            return new EncodedText(new double[dimensions()], 0);
        }

        return NativeRuntime.withLock(() -> {
            var inputs = CnnVocabulary.inputs(List.of(ids));
            double[] values;

            try {
                values = model.network().output(inputs.features(), false, inputs.mask(), null).toDoubleVector();
            } finally {
                model.network().clearLayerMaskArrays();
            }

            double norm = Math.sqrt(Arrays.stream(values).map(value -> value * value).sum());

            if (!Double.isFinite(norm)) {
                throw new IllegalStateException("CNN: неконечный вектор текста");
            }

            if (norm > 1e-12) {
                for (int i = 0; i < values.length; i++) {
                    values[i] /= norm;
                }
            } else {
                Arrays.fill(values, 0);
            }

            return new EncodedText(values, known);
        });
    }
}
