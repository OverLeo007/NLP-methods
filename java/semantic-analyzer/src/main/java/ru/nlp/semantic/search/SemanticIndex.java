package ru.nlp.semantic.search;

import java.util.List;
import java.util.Set;

import ru.nlp.semantic.search.dto.SearchHit;

/** Поиск по готовым нормализованным векторам, независимый от способа их получения. */
public interface SemanticIndex {

    /** Ранжирует ненулевые документы по нормализованному вектору запроса; для нулевого возвращает пустую выдачу. */
    List<SearchHit> search(double[] queryVector, int top);

    /** Возвращает число документов с нулевыми векторами. */
    long zeroDocuments();

    /** Возвращает ID документов без поискового вектора для диагностики пропусков. */
    Set<String> zeroDocumentIds();
}
