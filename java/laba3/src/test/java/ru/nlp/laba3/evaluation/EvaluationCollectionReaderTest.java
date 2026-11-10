package ru.nlp.laba3.evaluation;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import com.fasterxml.jackson.databind.node.ObjectNode;

import ru.nlp.laba3.TestComponents;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.evaluation.RelevanceJudgment;
import ru.nlp.laba3.model.evaluation.SubsetParameters;

class EvaluationCollectionReaderTest {

    @TempDir
    Path directory;

    @Test
    void readsExactlyThePreparedCollectionAndProtectsItsLists() throws Exception {
        var source = prepared();
        var data = reader().read(source);
        assertThat(data.documents()).hasSize(3);
        assertThat(data.queries()).extracting(EvaluationQuery::id).containsExactly("q");
        assertThat(data.sha256()).hasSize(4);
        assertThat(data.judgments().get("q")).extracting(RelevanceJudgment::documentId).containsExactly("a");
        assertThatThrownBy(() -> data.documents().clear()).isInstanceOf(UnsupportedOperationException.class);
        assertThatThrownBy(() -> data.judgments().get("q").clear()).isInstanceOf(UnsupportedOperationException.class);
    }

    @ParameterizedTest
    @ValueSource(strings = {"corpus.csv", "queries.jsonl", "qrels.jsonl"})
    void rejectsModifiedDataBeforeEvaluation(String name) throws Exception {
        var source = prepared();
        Files.writeString(source.resolve(name), "tampered");
        assertThatThrownBy(() -> reader().read(source)).isInstanceOf(InvalidInputException.class)
                .hasMessageContaining("SHA-256");
    }

    @ParameterizedTest
    @ValueSource(strings = {"profile", "ids", "queries", "missing-relation", "missing-document"})
    void rejectsInconsistentCollectionEvenWithUpdatedChecksums(String fault) throws Exception {
        var source = prepared();
        var mapper = new JsonConfiguration().jsonMapper();
        var manifestPath = source.resolve("manifest.json");
        var manifest = (ObjectNode) mapper.readTree(manifestPath.toFile());

        switch (fault) {
            case "profile" -> manifest.put("preprocessing", "wrong");
            case "ids" -> manifest.putArray("documentIds").add("wrong");
            case "queries" -> manifest.putArray("queryIds").add("wrong");
            case "missing-relation" -> Files.writeString(source.resolve("qrels.jsonl"), "");
            case "missing-document" -> Files.writeString(
                    source.resolve("qrels.jsonl"),
                    mapper.writeValueAsString(new RelevanceJudgment("q", "absent", 1)) + "\n"
            );
            default -> throw new AssertionError(fault);
        }

        ((ObjectNode) manifest.get("outputSha256"))
                .put("qrels.jsonl", FileChecksums.hash(source.resolve("qrels.jsonl")));
        mapper.writeValue(manifestPath.toFile(), manifest);
        assertThatThrownBy(() -> reader().read(source)).isInstanceOf(InvalidInputException.class);
    }

    private EvaluationCollectionReader reader() throws Exception {
        var mapper = new JsonConfiguration().jsonMapper();

        return new EvaluationCollectionReader(
                mapper,
                TestComponents.corpusReader(),
                new EvaluationDataReader(mapper),
                TestComponents.preprocessor()
        );
    }

    private Path prepared() throws Exception {
        var configuration = new JsonConfiguration();
        var mapper = configuration.jsonMapper();
        var corpus = directory.resolve("corpus.jsonl");
        var queries = directory.resolve("queries.jsonl");
        var qrels = directory.resolve("qrels.jsonl");
        var docs = List.of(
                Map.of("_id", "a", "title", "", "text", "Ракеты спутники космос"),
                Map.of("_id", "b", "title", "", "text", "Футбол матчи команды"),
                Map.of("_id", "c", "title", "", "text", "Музыка концерты песни")
        );
        Files.write(corpus, docs.stream().map(doc -> {
            try {
                return mapper.writeValueAsString(doc);
            } catch (Exception exception) {
                throw new AssertionError(exception);
            }
        }).toList());
        Files.writeString(queries, mapper.writeValueAsString(new EvaluationQuery("q", "Ракеты")) + "\n");
        Files.writeString(qrels, mapper.writeValueAsString(new RelevanceJudgment("q", "a", 1)) + "\n");
        var output = directory.resolve("prepared");
        new RiaSubsetPreparation(
                mapper,
                configuration.csvMapper(),
                TestComponents.corpusReader(),
                TestComponents.preprocessor(),
                new EvaluationDataReader(mapper)
        )
                .prepare(corpus, queries, qrels, output, new SubsetParameters(3, 1, 42));

        return output;
    }
}
