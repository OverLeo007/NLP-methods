package ru.nlp.laba3.search;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.search.SearchHit;

/** Неизменяемый индекс с полным перебором документов и cosine-ранжированием. */
public final class CosineIndex implements SemanticIndex {

    private final List<Entry> entries;

    private final int dimensions;

    private final long zeroDocuments;

    private final Set<String> zeroDocumentIds;

    /** Копирует готовые нормализованные векторы документов; обучение не выполняется. */
    public CosineIndex(List<Document> documents, List<double[]> vectors, int dimensions) {
        if (dimensions < 1 || documents.size() != vectors.size()) {
            throw new InvalidInputException("Размерность >= 1; каждому документу требуется один вектор");
        }
        this.dimensions = dimensions;
        var fittedEntries = new ArrayList<Entry>();
        long zeros = 0;
        var zeroIds = new HashSet<String>();

        for (int i = 0; i < documents.size(); i++) {
            double[] vector = vectors.get(i).clone();
            validateVector(vector);

            if (isZero(vector)) {
                zeros++;
                zeroIds.add(documents.get(i).id());
            } else {
                fittedEntries.add(new Entry(documents.get(i), vector));
            }
        }

        entries = List.copyOf(fittedEntries);
        zeroDocuments = zeros;
        zeroDocumentIds = Set.copyOf(zeroIds);
    }

    /** Считает cosine как скалярное произведение L2-векторов; равенства разрешаются по ID. */
    @Override
    public List<SearchHit> search(double[] queryVector, int top) {
        if (top < 1) {
            throw new InvalidInputException("top должен быть >= 1");
        }
        validateVector(queryVector);

        if (isZero(queryVector)) {
            return List.of();
        }
        var hits = new ArrayList<SearchHit>();

        for (var entry : entries) {
            double similarity = 0;

            for (int i = 0; i < dimensions; i++) {
                similarity += queryVector[i] * entry.values()[i];
            }

            var document = entry.document();
            String excerpt = document.text().replaceAll("\\s+", " ");
            hits.add(
                    new SearchHit(
                            document.id(),
                            document.title(),
                            Math.clamp(similarity, -1, 1),
                            excerpt.substring(0, Math.min(180, excerpt.length()))
                    )
            );
        }

        hits.sort(Comparator.comparingDouble(SearchHit::cosine).reversed().thenComparing(SearchHit::id));

        return List.copyOf(hits.subList(0, Math.min(top, hits.size())));
    }

    /** Возвращает число исключённых нулевых документов. */
    @Override
    public long zeroDocuments() {
        return zeroDocuments;
    }

    /** Возвращает неизменяемые ID исключённых нулевых документов. */
    @Override
    public Set<String> zeroDocumentIds() {
        return zeroDocumentIds;
    }

    private void validateVector(double[] vector) {
        if (vector.length != dimensions) {
            throw new InvalidInputException("Размерность вектора не соответствует индексу");
        }

        for (double value : vector) {
            if (!Double.isFinite(value)) {
                throw new InvalidInputException("Координаты вектора должны быть конечными");
            }
        }
    }

    private static boolean isZero(double[] vector) {
        for (double value : vector) {
            if (value != 0) {
                return false;
            }
        }

        return true;
    }

    private record Entry(
            /** Документ индекса. */
            Document document,
            /** Собственная копия координат документа. */
            double[] values
    ) {}
}
