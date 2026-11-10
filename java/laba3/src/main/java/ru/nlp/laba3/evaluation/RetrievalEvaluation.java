package ru.nlp.laba3.evaluation;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.backend.ModelBackend;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.evaluation.EvaluationCollection;
import ru.nlp.laba3.model.evaluation.EvaluationReport;
import ru.nlp.laba3.model.evaluation.EvaluationSummary;
import ru.nlp.laba3.model.evaluation.ModelEvaluation;
import ru.nlp.laba3.model.evaluation.QueryEvaluation;
import ru.nlp.laba3.model.evaluation.RelevanceJudgment;
import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.model.word2vec.Word2VecParameters;
import ru.nlp.laba3.runtime.NativeRuntime;
import ru.nlp.laba3.runtime.RuntimeMetadata;
import ru.nlp.laba3.runtime.StageTimings;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Одинаковый оценочный сценарий над двумя сохранёнными моделями без переобучения. */
@Service
@RequiredArgsConstructor
@Slf4j
public class RetrievalEvaluation {

    private final EvaluationCollectionReader collectionReader;

    private final ModelBackend<LsaParameters> lsaBackend;

    private final ModelBackend<Word2VecParameters> word2VecBackend;

    private final ObjectMapper jsonMapper;

    private final RuntimeMetadata runtimeMetadata;

    /** Проверяет коллекцию, оценивает обе модели и сохраняет новый подробный протокол. */
    public EvaluationSummary evaluate(
            Path collectionDirectory,
            Path lsaModel,
            Path word2VecModel,
            Path output
    ) throws IOException {
        if (Files.exists(output)) {
            throw new InvalidInputException("Каталог оценки уже существует: " + output);
        }
        var collection = collectionReader.read(collectionDirectory);
        var lsa = evaluateModel(Algorithm.lsa, lsaModel, lsaBackend, collection);
        var word2vec = evaluateModel(Algorithm.word2vec, word2VecModel, word2VecBackend, collection);

        for (var entry : collection.sha256().entrySet()) {
            if (!entry.getValue().equals(FileChecksums.hash(collectionDirectory.resolve(entry.getKey())))) {
                throw new InvalidInputException("Коллекция изменена во время оценки: " + entry.getKey());
            }
        }

        var versions = new TreeMap<>(NativeRuntime.withThreads(1, runtimeMetadata::versions));
        versions.put("maxHeapBytes", Long.toString(Runtime.getRuntime().maxMemory()));
        versions.put("jvmArguments", ManagementFactory.getRuntimeMXBean().getInputArguments().toString());
        var report = new EvaluationReport(
                "ria-recall5-mrr10-all-queries-v1",
                collection.manifest(),
                collection.sha256(),
                versions,
                "Sequential LSA then Word2Vec; no warmup; opening includes reading/checks/indexing; "
                        + "each query timed once with normalization/vectorization/full ranking/excerpts; "
                        + "hashing and report writing excluded; p95=nearest rank; all queries in denominator.",
                List.of(lsa, word2vec)
        );
        Files.createDirectories(output.toAbsolutePath().getParent());
        Files.createDirectory(output);
        Files.writeString(
                output.resolve("report.json"),
                jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(report) + "\n",
                StandardCharsets.UTF_8
        );
        log.info("[Оценка] Протокол сохранён: {}", output);

        return new EvaluationSummary(
                output.toString(),
                collection.queries().size(),
                List.of(
                        new EvaluationSummary.ModelSummary(lsa.algorithm(), lsa.zeroDocuments(), lsa.metrics()),
                        new EvaluationSummary.ModelSummary(
                                word2vec.algorithm(),
                                word2vec.zeroDocuments(),
                                word2vec.metrics()
                        )
                )
        );
    }

    private <P> ModelEvaluation<P> evaluateModel(
            Algorithm algorithm,
            Path directory,
            ModelBackend<P> backend,
            EvaluationCollection collection
    ) throws IOException {
        var hashes = modelHashes(directory);
        log.info("[{}] Оценка: открытие модели {}", algorithm.displayName(), directory);
        long started = System.nanoTime();
        var opened = backend.openModel(directory);
        double openingSeconds = StageTimings.elapsed(started);
        var session = opened.session();

        if (
            !session.documents().equals(collection.documents())
                    || !opened.metadata().normalization().equals(collection.manifest().preprocessing())
                    || !opened.metadata().stopwordsSha256().equals(collection.manifest().stopwordsSha256())
        ) {
            throw new InvalidInputException(
                    algorithm.displayName()
                            + ": модель обучена на другом составе/порядке/тексте документов или профиле"
            );
        }
        var results = new ArrayList<QueryEvaluation>();

        for (var query : collection.queries()) {
            started = System.nanoTime();
            var response = session.search(query.text(), 10);
            double seconds = StageTimings.elapsed(started);
            var relevantIds = collection.judgments().get(query.id()).stream()
                    .map(RelevanceJudgment::documentId).collect(Collectors.toSet());
            results.add(
                    RetrievalMetricsCalculator.score(query, relevantIds, session.zeroDocumentIds(), response, seconds)
            );

            if (results.size() % 100 == 0 || results.size() == collection.queries().size()) {
                log.info(
                        "[{}] Оценено запросов={}/{}",
                        algorithm.displayName(),
                        results.size(),
                        collection.queries().size()
                );
            }
        }

        if (!hashes.equals(modelHashes(directory))) {
            throw new InvalidInputException("Модель изменена во время оценки: " + directory);
        }

        return new ModelEvaluation<>(
                algorithm,
                directory.toString(),
                hashes,
                opened.metadata(),
                session.documents().size(),
                session.zeroDocuments(),
                session.zeroDocumentIds().stream().sorted().toList(),
                openingSeconds,
                RetrievalMetricsCalculator.aggregate(results),
                results
        );
    }

    private Map<String, String> modelHashes(Path directory) throws IOException {
        var hashes = new TreeMap<String, String>();

        try (var paths = Files.list(directory)) {
            for (var path : paths.filter(Files::isRegularFile).sorted().toList()) {
                hashes.put(path.getFileName().toString(), FileChecksums.hash(path));
            }
        }

        return hashes;
    }
}
