package ru.nlp.laba3.search;

import java.util.List;
import java.util.Set;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.search.SearchResult;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.vectorization.TextVectorizer;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;

/** Загруженная поисковая сессия: общая обработка текста, векторизация и индекс. */
@RequiredArgsConstructor(access = AccessLevel.PRIVATE)
public final class SearchSession {

    private final List<Document> documents;

    private final TextPreprocessor preprocessor;

    private final TextVectorizer vectorizer;

    private final SemanticIndex index;

    /** Строит общий cosine-индекс через переданный векторизатор любого backend. */
    public static SearchSession create(
            List<Document> documents,
            TextPreprocessor preprocessor,
            TextVectorizer vectorizer
    ) {
        var vectors = documents.stream()
                .map(document -> vectorizer.encode(preprocessor.tokens(document.text())).values()).toList();

        return new SearchSession(
                List.copyOf(documents),
                preprocessor,
                vectorizer,
                new CosineIndex(documents, vectors, vectorizer.dimensions())
        );
    }

    /** Обрабатывает запрос без повторной загрузки модели и построения индекса. */
    public SearchResult search(String text, int top) {
        if (top < 1) {
            throw new InvalidInputException("top должен быть >= 1");
        }
        var tokens = preprocessor.tokens(text);

        if (tokens.isEmpty()) {
            return new SearchResult("empty_after_normalization", 0, 0, List.of());
        }
        var query = vectorizer.encode(tokens);

        if (query.knownTokens() == 0 || query.isZero()) {
            return new SearchResult(
                    query.knownTokens() == 0 ? "oov" : "zero_vector",
                    query.knownTokens(),
                    tokens.size(),
                    List.of()
            );
        }

        return new SearchResult("ok", query.knownTokens(), tokens.size(), index.search(query.values(), top));
    }

    /** Возвращает число документов, исключённых из выдачи из-за нулевого вектора. */
    public long zeroDocuments() {
        return index.zeroDocuments();
    }

    /** Возвращает неизменяемый полный корпус, включая нулевые документы. */
    public List<Document> documents() {
        return documents;
    }

    /** Возвращает ID нулевых документов для оценки причин пропусков. */
    public Set<String> zeroDocumentIds() {
        return index.zeroDocumentIds();
    }
}
