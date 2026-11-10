package ru.nlp.laba3.search;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.search.SearchHit;

class CosineIndexTest {

    @Test
    void ranksVectorsIncludingNegativeCosineAndTiesWithoutZeroDocuments() {
        var documents = List.of(
                document("b"),
                document("a"),
                document("orthogonal"),
                document("opposite"),
                document("zero")
        );
        var vectors = List.of(
                new double[]{1, 0},
                new double[]{1, 0},
                new double[]{0, 1},
                new double[]{-1, 0},
                new double[]{0, 0}
        );
        SemanticIndex index = new CosineIndex(documents, vectors, 2);
        var hits = index.search(new double[]{1, 0}, 20);
        assertThat(hits).extracting(SearchHit::id).containsExactly("a", "b", "orthogonal", "opposite");
        assertThat(hits).extracting(SearchHit::cosine).containsExactly(1.0, 1.0, 0.0, -1.0);
        assertThat(index.search(new double[]{1, 0}, 2)).isEqualTo(hits.subList(0, 2));
        assertThat(index.search(new double[]{0, 0}, 2)).isEmpty();
        assertThat(index.zeroDocuments()).isEqualTo(1);
        assertThat(index.zeroDocumentIds()).containsExactly("zero");
        assertThatThrownBy(index.zeroDocumentIds()::clear).isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(hits::clear).isInstanceOf(UnsupportedOperationException.class);
    }

    @Test
    void ownsDocumentVectorsAndRejectsMalformedCoordinates() {
        var vector = new double[]{1, 0};
        var documents = new ArrayList<>(List.of(document("a")));
        var vectors = new ArrayList<>(List.of(vector));
        var index = new CosineIndex(documents, vectors, 2);
        vector[0] = 0;
        documents.clear();
        vectors.clear();
        assertThat(index.search(new double[]{1, 0}, 1).getFirst().cosine()).isEqualTo(1);
        assertThatThrownBy(() -> index.search(new double[]{1}, 1)).hasMessageContaining("Размерность");
        assertThatThrownBy(() -> index.search(new double[]{Double.NaN, 0}, 1)).hasMessageContaining("конечными");
        assertThatThrownBy(() -> index.search(new double[]{1, 0}, 0)).hasMessageContaining("top");
        assertThatThrownBy(() -> new CosineIndex(List.of(document("a")), List.of(), 2))
                .hasMessageContaining("один вектор");
        assertThatThrownBy(
                () -> new CosineIndex(
                        List.of(document("a")),
                        List.of(new double[]{1, Double.POSITIVE_INFINITY}),
                        2
                )
        ).hasMessageContaining("конечными");
    }

    private static Document document(String id) {
        return new Document(id, id, "Текст документа " + id);
    }
}
