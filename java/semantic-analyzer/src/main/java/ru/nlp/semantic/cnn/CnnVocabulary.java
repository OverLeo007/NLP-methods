package ru.nlp.semantic.cnn;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import ru.nlp.semantic.exception.InvalidInputException;

/** Детерминированный словарь CNN: PAD=0, OOV=1, обычные слова начинаются с 2. */
public final class CnnVocabulary {

    private final List<String> words;

    private final Map<String, Integer> positions;

    /** Восстанавливает порядок ID и проверяет словарь без служебных значений. */
    public CnnVocabulary(List<String> words) {
        this.words = List.copyOf(words);
        var indices = new HashMap<String, Integer>();

        for (int i = 0; i < words.size(); i++) {
            String word = words.get(i);

            if (word.isBlank() || indices.put(word, i + 2) != null) {
                throw new InvalidInputException("CNN: словарь содержит пустые или повторяющиеся термины");
            }
        }

        if (words.size() < 2) {
            throw new InvalidInputException("CNN: после min-count нужны минимум два термина");
        }
        positions = Map.copyOf(indices);
    }

    /** Строит словарь только по переданным обучающим документам, без запросов и меток. */
    public static CnnVocabulary fit(List<List<String>> documents, int minCount, int maxFeatures) {
        var frequencies = new HashMap<String, Integer>();
        documents.forEach(document -> document.forEach(word -> frequencies.merge(word, 1, Integer::sum)));
        var words = frequencies.keySet().stream().filter(word -> frequencies.get(word) >= minCount)
                .sorted(
                        Comparator.<String>comparingInt(frequencies::get).reversed()
                                .thenComparing(Comparator.naturalOrder())
                )
                .limit(maxFeatures).toList();

        return new CnnVocabulary(words);
    }

    /** Возвращает неизменяемый словарь в порядке ID. */
    public List<String> words() {
        return words;
    }

    /** Переводит токены в ID с сохранением позиций неизвестных слов. */
    public int[] encode(List<String> tokens) {
        return tokens.stream().mapToInt(token -> positions.getOrDefault(token, 1)).toArray();
    }

    /** Дополняет пакет до его максимальной длины; PAD не участвует в свёртке и pooling. */
    public static Inputs inputs(List<int[]> rows) {
        int length = rows.stream().mapToInt(row -> row.length).max().orElseThrow();

        if (length == 0 || rows.stream().anyMatch(row -> row.length == 0)) {
            throw new IllegalArgumentException("CNN: пустые строки не передаются в сеть");
        }
        var features = Nd4j.zeros(DataType.FLOAT, rows.size(), length);
        var mask = Nd4j.zeros(DataType.FLOAT, rows.size(), length);

        for (int i = 0; i < rows.size(); i++) {
            for (int j = 0; j < rows.get(i).length; j++) {
                features.putScalar(i, j, rows.get(i)[j]);
                mask.putScalar(i, j, 1);
            }
        }

        return new Inputs(features, mask);
    }

    /** Считает известные вхождения среди ID, фактически переданных кодировщику. */
    public static int knownTokens(int[] ids) {
        return (int) Arrays.stream(ids).filter(id -> id >= 2).count();
    }

    /** Входы сети; маска относится к последовательности, а не к функции потерь. */
    public record Inputs(
            /** Индексы слов [пакет, длина]. */
            INDArray features,
            /** Единицы для реальных позиций и нули для PAD. */
            INDArray mask
    ) {}
}
