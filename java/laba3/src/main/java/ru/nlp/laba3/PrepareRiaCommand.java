package ru.nlp.laba3;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.evaluation.RiaSubsetPreparation;
import ru.nlp.laba3.model.evaluation.SubsetParameters;

import lombok.RequiredArgsConstructor;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Spec;

/** CLI подготовки коллекции РИА, отдельный от операций над моделями. */
@Component
@Command(
        name = "prepare-ria",
        mixinStandardHelpOptions = true,
        showDefaultValues = true,
        description = "Подготовить общую коллекцию РИА с сохранением всех релевантных статей."
)
@RequiredArgsConstructor
public class PrepareRiaCommand implements Callable<Integer> {

    private final RiaSubsetPreparation preparation;

    private final ObjectMapper jsonMapper;

    @Spec
    private CommandSpec specification;

    @Option(names = "--corpus", required = true, description = "Исходный corpus.jsonl.zip или JSONL РИА.")
    private Path corpus;

    @Option(names = "--queries", required = true, description = "Исходный queries.jsonl.")
    private Path queries;

    @Option(names = "--qrels", required = true, description = "Исходный test.jsonl с разметкой релевантности.")
    private Path qrels;

    @Option(names = "--output", required = true, description = "Новый каталог для коллекции и манифеста.")
    private Path output;

    @Option(names = "--documents", defaultValue = "10000", description = "Число уникальных непустых статей.")
    private int documents;

    @Option(names = "--query-count", defaultValue = "500", description = "Число пригодных оценочных запросов.")
    private int queryCount;

    @Option(names = "--seed", defaultValue = "42", description = "Seed отбора запросов.")
    private long seed;

    /** Выводит итог отбора в JSON; прогресс включается унаследованным --verbose. */
    @Override
    public Integer call() throws Exception {
        var result = preparation
                .prepare(corpus, queries, qrels, output, new SubsetParameters(documents, queryCount, seed));
        specification.commandLine().getOut()
                .println(jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

        return 0;
    }
}
