package ru.nlp.laba3.evaluation;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeMap;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.corpus.reader.CorpusReader;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.corpus.Document;
import ru.nlp.laba3.model.evaluation.EvaluationCollection;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.evaluation.RiaSubsetManifest;
import ru.nlp.laba3.preprocessing.TextPreprocessor;

import lombok.RequiredArgsConstructor;

/** Проверяет происхождение и полный состав коллекции до открытия моделей. */
@Component
@RequiredArgsConstructor
public class EvaluationCollectionReader {

    private final ObjectMapper jsonMapper;

    private final CorpusReader corpusReader;

    private final EvaluationDataReader dataReader;

    private final TextPreprocessor preprocessor;

    /** Читает подготовленный каталог без исключения запросов по покрытию модели. */
    public EvaluationCollection read(Path directory) throws IOException {
        var manifest = jsonMapper.readValue(directory.resolve("manifest.json").toFile(), RiaSubsetManifest.class);

        if (
            !"ria-known-collection-held-out-queries-v1".equals(manifest.protocol())
                    || manifest.parameters() == null
                    || !preprocessor.profile().equals(manifest.preprocessing())
                    || !preprocessor.resourceHash().equals(manifest.stopwordsSha256())
        ) {
            throw new InvalidInputException("Несовместимый манифест оценочной коллекции");
        }
        var hashes = new TreeMap<String, String>();

        for (String name : new String[]{"corpus.csv", "queries.jsonl", "qrels.jsonl"}) {
            String hash = FileChecksums.hash(directory.resolve(name));

            if (!hash.equals(manifest.outputSha256().get(name))) {
                throw new InvalidInputException("Не совпадает SHA-256 файла коллекции: " + name);
            }
            hashes.put(name, hash);
        }

        hashes.put("manifest.json", FileChecksums.hash(directory.resolve("manifest.json")));
        var documents = new ArrayList<Document>();
        corpusReader.scan(directory.resolve("corpus.csv"), document -> {
            documents.add(document);

            return true;
        });
        var ids = new HashSet<String>();

        for (var document : documents) {
            if (
                document.id() == null || document.id().isBlank() || document.text() == null
                        || document.text().isBlank() || !ids.add(document.id())
            ) {
                throw new InvalidInputException("Пустой или повторный документ оценочной коллекции");
            }
        }

        var queries = dataReader.readQueries(directory.resolve("queries.jsonl"));

        if (
            documents.size() != manifest.parameters().documents()
                    || queries.size() != manifest.parameters().queries()
                    || !documents.stream().map(Document::id).toList().equals(manifest.documentIds())
                    || !queries.stream().map(EvaluationQuery::id).toList().equals(manifest.queryIds())
        ) {
            throw new InvalidInputException("Состав коллекции не совпадает с манифестом");
        }
        var judgments = dataReader.readJudgments(directory.resolve("qrels.jsonl"), queries);

        for (var query : queries) {
            var relations = judgments.getOrDefault(query.id(), java.util.List.of());

            if (relations.isEmpty() || relations.stream().anyMatch(row -> !ids.contains(row.documentId()))) {
                throw new InvalidInputException("Нет полной положительной разметки запроса: " + query.id());
            }
        }

        return new EvaluationCollection(manifest, documents, queries, judgments, hashes);
    }
}
