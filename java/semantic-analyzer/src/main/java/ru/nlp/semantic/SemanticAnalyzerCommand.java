package ru.nlp.semantic;

import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.semantic.backend.BackendRegistry;
import ru.nlp.semantic.backend.ModelBackend;
import ru.nlp.semantic.cnn.PairSampling;
import ru.nlp.semantic.exception.InvalidInputException;
import ru.nlp.semantic.model.analysis.Algorithm;
import ru.nlp.semantic.model.cnn.CnnParameters;
import ru.nlp.semantic.model.lsa.LsaParameters;
import ru.nlp.semantic.model.training.TrainingSummary;
import ru.nlp.semantic.model.word2vec.Word2VecParameters;
import ru.nlp.semantic.preprocessing.TextPreprocessor;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.ScopeType;
import picocli.CommandLine.Spec;

/** Корневая команда нормализации, обучения и поиска. */
@Component
@Command(
        name = "semantic-analyzer",
        mixinStandardHelpOptions = true,
        description = "Семантический анализ документов: Word2Vec, LSA и CNN.",
        subcommands = {SemanticAnalyzerCommand.TrainCommand.class, SemanticAnalyzerCommand.SearchCommand.class,
                PrepareRiaCommand.class,
                EvaluateCommand.class},
        version = "SemanticAnalyzer 0.1.0"
)
@RequiredArgsConstructor
public class SemanticAnalyzerCommand implements Callable<Integer> {

    private final TextPreprocessor preprocessor;

    @Spec
    private CommandSpec specification;

    /** Показывает, включён ли подробный вывод для текущего вызова CLI. */
    @Getter
    @Option(
            names = {"-v", "--verbose"},
            scope = ScopeType.INHERIT,
            description = "Включить логи и прогресс в stderr."
    )
    private boolean verbose;

    @Option(names = "--text", description = "Текст для проверки нормализации и Snowball.")
    private String text;

    /** Показывает нормализованные токены или справку по командам. */
    @Override
    public Integer call() {

        if (text == null) {
            specification.commandLine().usage(specification.commandLine().getOut());
        } else {
            preprocessor.tokens(text).forEach(specification.commandLine().getOut()::println);
        }

        return 0;
    }

    /** Обучение выбранного анализатора на корпусе. */
    @Component
    @Command(
            name = "train",
            mixinStandardHelpOptions = true,
            showDefaultValues = true,
            description = "Обучить выбранный анализатор и сохранить модель в новый каталог."
    )
    @RequiredArgsConstructor
    public static class TrainCommand implements Callable<Integer> {

        private final ModelBackend<Word2VecParameters> word2VecBackend;

        private final ModelBackend<LsaParameters> lsaBackend;

        private final ModelBackend<CnnParameters> cnnBackend;

        private final ObjectMapper jsonMapper;

        @Spec
        private CommandSpec specification;

        @Option(names = "--input", required = true, description = "ZIP РИА, JSONL, CSV или папка TXT.")
        private Path input;

        @Option(names = "--output", required = true, description = "Новый каталог для модели.")
        private Path output;

        @Option(names = "--algorithm", defaultValue = "word2vec", description = "Алгоритм: ${COMPLETION-CANDIDATES}.")
        private Algorithm algorithm;

        @Option(names = "--max-features", defaultValue = "20000", description = "Лимит словаря LSA/CNN по частоте.")
        private int maxFeatures;

        @Option(names = "--limit", defaultValue = "1000", description = "Лимит уникальных документов.")
        private int limit;

        @Option(
                names = "--dimensions",
                defaultValue = "50",
                description = "Размерность embeddings или число LSA-компонент."
        )
        private int dimensions;

        @Option(names = "--epochs", defaultValue = "3", description = "Количество эпох Word2Vec/CNN.")
        private int epochs;

        @Option(names = "--min-count", defaultValue = "2", description = "Минимальная частота слова Word2Vec/CNN.")
        private int minCount;

        @Option(names = "--window", defaultValue = "5", description = "Окно Word2Vec.")
        private int window;

        @Option(names = "--seed", defaultValue = "42", description = "Seed обучения Word2Vec/CNN.")
        private long seed;

        @Option(names = "--workers", defaultValue = "1", description = "Потоки обучения и ND4J.")
        private int workers;

        @Mixin
        private final CnnOptions cnnOptions = new CnnOptions();

        /** Выводит итог обучения в JSON; сообщения прогресса идут в stderr. */
        @Override
        public Integer call() throws Exception {
            requireAlgorithms(Set.of(Algorithm.word2vec, Algorithm.cnn), "--epochs", "--min-count", "--seed");
            requireAlgorithms(Set.of(Algorithm.lsa, Algorithm.cnn), "--max-features");
            requireAlgorithms(Set.of(Algorithm.word2vec), "--window");
            requireAlgorithms(Set.of(Algorithm.cnn), CnnOptions.NAMES);
            // Композиция типизированных параметров находится только на границе CLI.
            TrainingSummary result = switch (algorithm) {
                case lsa -> lsaBackend.train(input, output, limit, new LsaParameters(dimensions, maxFeatures, workers));
                case word2vec -> word2VecBackend.train(
                        input,
                        output,
                        limit,
                        new Word2VecParameters(dimensions, epochs, minCount, window, seed, workers)
                );
                case cnn -> cnnBackend.train(
                        input,
                        output,
                        limit,
                        cnnOptions.parameters(dimensions, maxFeatures, minCount, epochs, seed, workers)
                );
            };
            specification
                    .commandLine()
                    .getOut()
                    .println(jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

            return 0;
        }

        private void requireAlgorithms(Set<Algorithm> allowed, String... options) {
            for (String option : options) {
                if (
                    !allowed.contains(algorithm)
                            && specification.commandLine().getParseResult().hasMatchedOption(option)
                ) {
                    throw new InvalidInputException(
                            option + " относится только к " + allowed.stream().sorted()
                                    .map(Algorithm::displayName).collect(Collectors.joining(", "))
                    );
                }
            }
        }
    }

    private static final class CnnOptions {

        private static final String[] NAMES = {"--embedding-dimensions", "--filters", "--kernel-size",
                "--max-length", "--batch-size", "--learning-rate", "--temperature", "--validation-fraction",
                "--pair-sampling"};

        @Option(names = "--embedding-dimensions", defaultValue = "32", description = "CNN: размер вектора слова.")
        private int embeddingDimensions;

        @Option(names = "--filters", defaultValue = "32", description = "CNN: число свёрточных фильтров.")
        private int filters;

        @Option(names = "--kernel-size", defaultValue = "3", description = "CNN: нечётная ширина свёртки.")
        private int kernelSize;

        @Option(names = "--max-length", defaultValue = "96", description = "CNN: предел токенов фрагмента и поиска.")
        private int maxLength;

        @Option(
                names = "--batch-size",
                defaultValue = "16",
                description = "CNN: документов в пакете (по два фрагмента)."
        )
        private int batchSize;

        @Option(names = "--learning-rate", defaultValue = "0.003", description = "CNN: скорость Adam.")
        private double learningRate;

        @Option(names = "--temperature", defaultValue = "0.1", description = "CNN: температура NT-Xent.")
        private double temperature;

        @Option(
                names = "--validation-fraction",
                defaultValue = "0",
                description = "CNN: доля отложенных документов для выбора эпохи, от 0 до 0.5."
        )
        private double validationFraction;

        @Option(
                names = "--pair-sampling",
                defaultValue = "symmetric",
                description = "CNN: выбор положительных пар: ${COMPLETION-CANDIDATES}."
        )
        private PairSampling pairSampling;

        private CnnParameters parameters(
                int dimensions,
                int maxFeatures,
                int minCount,
                int epochs,
                long seed,
                int workers
        ) {
            return CnnParameters.builder()
                    .dimensions(dimensions)
                    .embeddingDimensions(embeddingDimensions)
                    .filters(filters)
                    .kernelSize(kernelSize)
                    .maxLength(maxLength)
                    .maxFeatures(maxFeatures)
                    .minCount(minCount)
                    .batchSize(batchSize)
                    .epochs(epochs)
                    .learningRate(learningRate)
                    .temperature(temperature)
                    .seed(seed)
                    .workers(workers)
                    .validationFraction(validationFraction)
                    .pairSampling(pairSampling)
                    .build();
        }
    }

    /** Поиск по сохранённой модели без повторного обучения. */
    @Component
    @Command(
            name = "search",
            mixinStandardHelpOptions = true,
            showDefaultValues = true,
            description = "Найти документы по текстовому запросу."
    )
    @RequiredArgsConstructor
    public static class SearchCommand implements Callable<Integer> {

        private final BackendRegistry backends;

        private final ObjectMapper jsonMapper;

        @Spec
        private CommandSpec specification;

        @Option(names = "--model", required = true, description = "Каталог сохранённой модели.")
        private Path model;

        @Option(names = "--algorithm", defaultValue = "word2vec", description = "Алгоритм: ${COMPLETION-CANDIDATES}.")
        private Algorithm algorithm;

        @Option(names = "--query", required = true, description = "Текст запроса.")
        private String query;

        @Option(names = "--top", defaultValue = "5", description = "Количество результатов.")
        private int top;

        /** Выводит результаты, cosine и покрытие запроса в JSON. */
        @Override
        public Integer call() throws Exception {
            var result = backends.require(algorithm).open(model).search(query, top);
            specification
                    .commandLine()
                    .getOut()
                    .println(jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

            return 0;
        }
    }

}
