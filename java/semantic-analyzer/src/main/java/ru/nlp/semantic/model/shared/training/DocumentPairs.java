package ru.nlp.semantic.model.shared.training;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import ru.nlp.semantic.exception.InvalidInputException;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;

/** Выбор двух фрагментов документа без меток, с исключением точных отрицательных совпадений. */
public final class DocumentPairs {

    private final List<int[]> documents;

    private final int maxLength;

    private final int batchSize;

    private final PairSampling sampling;

    /** Убирает пустые/OOV документы и дубликаты последовательностей ID до выбора отрицательных пар. */
    public DocumentPairs(List<int[]> documents, int maxLength, int batchSize) {
        this(documents, maxLength, batchSize, PairSampling.symmetric);
    }

    /** Сохраняет способ выбора положительных пар; правила отрицательных пар общие. */
    public DocumentPairs(List<int[]> documents, int maxLength, int batchSize, PairSampling sampling) {
        var unique = new HashSet<List<Integer>>();
        this.documents = documents.stream().filter(ids -> SequenceVocabulary.knownTokens(ids) > 0)
                .filter(ids -> unique.add(key(ids))).map(int[]::clone).toList();
        this.maxLength = maxLength;
        this.batchSize = batchSize;
        this.sampling = Objects.requireNonNull(sampling);

        if (this.documents.size() < 2) {
            throw new InvalidInputException("Нужны минимум два разных непустых документа с известными токенами");
        }
    }

    /** Возвращает число различных документов, пригодных для выбора фрагментов. */
    public int documents() {
        return documents.size();
    }

    /** Создаёт воспроизводимую эпоху; финальный пакет из одного документа пропускается. */
    public List<Batch> batches(Random random) {
        var order = new ArrayList<Integer>();

        for (int i = 0; i < documents.size(); i++) {
            order.add(i);
        }

        Collections.shuffle(order, random);
        var batches = new ArrayList<Batch>();
        var rows = new ArrayList<int[]>();
        var sources = new ArrayList<Integer>();
        var seen = new HashSet<List<Integer>>();

        for (int source : order) {
            int[] first = null;
            int[] second = null;

            for (int attempt = 0; attempt < 32; attempt++) {
                var fragments = sampling.sample(documents.get(source), maxLength, random);
                int[] left = fragments.first();
                int[] right = fragments.second();

                if (!seen.contains(key(left)) && !seen.contains(key(right))) {
                    first = left;
                    second = right;
                    break;
                }
            }

            if (first == null) {
                continue;
            }
            rows.add(first);
            rows.add(second);
            sources.add(source);
            sources.add(source);
            seen.add(key(first));
            seen.add(key(second));

            if (rows.size() == batchSize * 2) {
                batches.add(new Batch(List.copyOf(rows), List.copyOf(sources)));
                rows.clear();
                sources.clear();
                seen.clear();
            }
        }

        if (rows.size() >= 4) {
            batches.add(new Batch(List.copyOf(rows), List.copyOf(sources)));
        }

        if (batches.isEmpty()) {
            throw new InvalidInputException("Не удалось сформировать пакет из двух различных пар фрагментов");
        }

        return List.copyOf(batches);
    }

    private static List<Integer> key(int[] values) {
        return Arrays.stream(values).boxed().toList();
    }

    /** Положительные партнёры расположены в соседних строках; разные пары имеют разные источники. */
    public record Batch(
            /** Последовательности ID: две на каждый документ. */
            List<int[]> rows,
            /** Индексы исходных документов для проверки состава пар. */
            List<Integer> sources
    ) {

        /** Формирует пакет NT-Xent с маской токенов и индексом партнёра для каждой строки. */
        public DataSet dataSet() {
            var inputs = SequenceVocabulary.inputs(rows);
            var partners = Nd4j.zeros(DataType.FLOAT, rows.size(), 1);

            for (int i = 0; i < rows.size(); i++) {
                partners.putScalar(i, 0, i ^ 1);
            }

            return new DataSet(inputs.features(), partners, inputs.mask(), null);
        }
    }
}
