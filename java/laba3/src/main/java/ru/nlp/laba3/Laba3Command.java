package ru.nlp.laba3;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.backend.ModelBackend;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.model.training.TrainingSummary;
import ru.nlp.laba3.model.word2vec.Word2VecParameters;
import ru.nlp.laba3.preprocessing.TextPreprocessor;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.ScopeType;
import picocli.CommandLine.Spec;

/** Корневая команда нормализации, обучения и поиска. */
@Component
@Command(
        name = "laba3",
        mixinStandardHelpOptions = true,
        description = "Семантический анализ документов: Word2Vec и LSA.",
        subcommands = {Laba3Command.TrainCommand.class, Laba3Command.SearchCommand.class, PrepareRiaCommand.class,
                EvaluateCommand.class},
        version = "Laba3 0.1.0"
)
@RequiredArgsConstructor
public class Laba3Command implements Callable<Integer> {

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
            description = "Обучить Word2Vec или LSA и сохранить модель в новый каталог."
    )
    @RequiredArgsConstructor
    public static class TrainCommand implements Callable<Integer> {

        private final ModelBackend<Word2VecParameters> word2VecBackend;

        private final ModelBackend<LsaParameters> lsaBackend;

        private final ObjectMapper jsonMapper;

        @Spec
        private CommandSpec specification;

        @Option(names = "--input", required = true, description = "ZIP РИА, JSONL, CSV или папка TXT.")
        private Path input;

        @Option(names = "--output", required = true, description = "Новый каталог для модели.")
        private Path output;

        @Option(names = "--algorithm", defaultValue = "word2vec", description = "Алгоритм: ${COMPLETION-CANDIDATES}.")
        private Algorithm algorithm;

        @Option(names = "--max-features", defaultValue = "20000", description = "Лимит словаря LSA по частоте.")
        private int maxFeatures;

        @Option(names = "--limit", defaultValue = "1000", description = "Лимит уникальных документов.")
        private int limit;

        @Option(
                names = "--dimensions",
                defaultValue = "50",
                description = "Размерность embeddings или число LSA-компонент."
        )
        private int dimensions;

        @Option(names = "--epochs", defaultValue = "3", description = "Количество эпох Word2Vec.")
        private int epochs;

        @Option(names = "--min-count", defaultValue = "2", description = "Минимальная частота слова Word2Vec.")
        private int minCount;

        @Option(names = "--window", defaultValue = "5", description = "Окно Word2Vec.")
        private int window;

        @Option(names = "--seed", defaultValue = "42", description = "Seed обучения Word2Vec.")
        private long seed;

        @Option(names = "--workers", defaultValue = "1", description = "Потоки обучения и ND4J.")
        private int workers;

        /** Выводит итог обучения в JSON; сообщения прогресса идут в stderr. */
        @Override
        public Integer call() throws Exception {
            TrainingSummary result;

            if (algorithm == Algorithm.lsa) {
                if (
                    specification.commandLine().getParseResult().hasMatchedOption("--epochs")
                            || specification.commandLine().getParseResult().hasMatchedOption("--min-count")
                            || specification.commandLine().getParseResult().hasMatchedOption("--window")
                            || specification.commandLine().getParseResult().hasMatchedOption("--seed")
                ) {
                    throw new InvalidInputException(
                            "--epochs, --min-count, --window и --seed относятся только к Word2Vec"
                    );
                }
                result = lsaBackend.train(input, output, limit, new LsaParameters(dimensions, maxFeatures, workers));
            } else {
                if (specification.commandLine().getParseResult().hasMatchedOption("--max-features")) {
                    throw new InvalidInputException("--max-features относится только к LSA");
                }
                var parameters = Word2VecParameters.builder()
                        .dimensions(dimensions)
                        .epochs(epochs)
                        .minCount(minCount)
                        .window(window)
                        .seed(seed)
                        .workers(workers)
                        .build();
                result = word2VecBackend.train(input, output, limit, parameters);
            }
            specification
                    .commandLine()
                    .getOut()
                    .println(jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

            return 0;
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

        private final ModelBackend<Word2VecParameters> word2VecBackend;

        private final ModelBackend<LsaParameters> lsaBackend;

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
            var result = algorithm == Algorithm.lsa
                    ? lsaBackend.open(model).search(query, top)
                    : word2VecBackend.open(model).search(query, top);
            specification
                    .commandLine()
                    .getOut()
                    .println(jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

            return 0;
        }
    }

}
