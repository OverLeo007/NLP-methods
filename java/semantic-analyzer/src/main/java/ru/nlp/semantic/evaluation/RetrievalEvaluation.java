package ru.nlp.semantic.evaluation;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

import org.springframework.stereotype.Service;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.semantic.backend.BackendRegistry;
import ru.nlp.semantic.exception.InvalidInputException;
import ru.nlp.semantic.io.FileChecksums;
import ru.nlp.semantic.model.analysis.Algorithm;
import ru.nlp.semantic.model.evaluation.EvaluationCollection;
import ru.nlp.semantic.model.evaluation.EvaluationReport;
import ru.nlp.semantic.model.evaluation.EvaluationSummary;
import ru.nlp.semantic.model.evaluation.EvaluationTarget;
import ru.nlp.semantic.model.evaluation.ModelEvaluation;
import ru.nlp.semantic.model.evaluation.QueryEvaluation;
import ru.nlp.semantic.model.evaluation.RelevanceJudgment;
import ru.nlp.semantic.runtime.NativeRuntime;
import ru.nlp.semantic.runtime.RuntimeMetadata;
import ru.nlp.semantic.runtime.StageTimings;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Одинаковый оценочный сценарий над упорядоченным списком моделей без переобучения. */
@Service
@RequiredArgsConstructor
@Slf4j
public class RetrievalEvaluation {

    private final EvaluationCollectionReader collectionReader;

    private final BackendRegistry backends;

    private final ObjectMapper jsonMapper;

    private final RuntimeMetadata runtimeMetadata;

    /** Проверяет коллекцию, оценивает все переданные модели и сохраняет единый протокол. */
    public EvaluationSummary evaluate(
            Path collectionDirectory,
            List<EvaluationTarget> models,
            Path output
    ) throws IOException {
        if (Files.exists(output)) {
            throw new InvalidInputException("Каталог оценки уже существует: " + output);
        }
        var targets = List.copyOf(models);
        var algorithms = EnumSet.noneOf(Algorithm.class);

        if (targets.isEmpty()) {
            throw new InvalidInputException("Для оценки нужна хотя бы одна модель");
        }

        for (var target : targets) {
            if (!algorithms.add(target.algorithm())) {
                throw new InvalidInputException("Повторная модель алгоритма: " + target.algorithm());
            }
        }

        var collection = collectionReader.read(collectionDirectory);
        var evaluated = new ArrayList<ModelEvaluation<?>>();

        for (var target : targets) {
            evaluated.add(evaluateModel(target, collection));
        }

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
                "Sequential " + targets.stream().map(target -> target.algorithm().displayName())
                        .collect(Collectors.joining(" then "))
                        + "; no warmup; opening includes reading/checks/indexing; "
                        + "each query timed once with normalization/vectorization/full ranking/excerpts; "
                        + "hashing and report writing excluded; p95=nearest rank; all queries in denominator.",
                evaluated
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
                evaluated.stream().map(
                        model -> new EvaluationSummary.ModelSummary(
                                model.algorithm(),
                                model.zeroDocuments(),
                                model.metrics()
                        )
                ).toList()
        );
    }

    private ModelEvaluation<?> evaluateModel(EvaluationTarget target, EvaluationCollection collection)
            throws IOException {
        var algorithm = target.algorithm();
        var directory = target.directory();
        var hashes = modelHashes(directory);
        log.info("[{}] Оценка: открытие модели {}", algorithm.displayName(), directory);
        long started = System.nanoTime();
        var opened = backends.require(algorithm).openModel(directory);
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
