package ru.nlp.semantic.model.lstm.vectorization;

import java.util.Arrays;
import java.util.List;

import ru.nlp.semantic.model.base.dto.EncodedText;
import ru.nlp.semantic.model.base.vectorization.TextVectorizer;
import ru.nlp.semantic.model.lstm.LstmModel;
import ru.nlp.semantic.model.lstm.network.LstmNetwork;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;
import ru.nlp.semantic.runtime.NativeRuntime;

/** LSTM-векторизация ограниченного префикса текста с явными PAD/OOV и L2-нормализацией. */
public final class LstmVectorizer implements TextVectorizer {

    private final LstmModel model;

    private final SequenceVocabulary vocabulary;

    /** Связывает обученный кодировщик с его словарём и правилами длины. */
    public LstmVectorizer(LstmModel model) {
        this.model = model;
        vocabulary = new SequenceVocabulary(model.vocabulary());
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
        int known = SequenceVocabulary.knownTokens(ids);

        if (known == 0) {
            return new EncodedText(new double[dimensions()], 0);
        }

        return NativeRuntime.withLock(() -> {
            var inputs = SequenceVocabulary.inputs(List.of(ids));
            var features = inputs.features().reshape(1, 1, ids.length);
            double[] values = LstmNetwork.encode(model.network(), features, inputs.mask()).toDoubleVector();

            double norm = Math.sqrt(Arrays.stream(values).map(value -> value * value).sum());

            if (!Double.isFinite(norm)) {
                throw new IllegalStateException("LSTM: неконечный вектор текста");
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
