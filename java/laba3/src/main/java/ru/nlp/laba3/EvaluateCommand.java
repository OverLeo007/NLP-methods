package ru.nlp.laba3;

import java.nio.file.Path;
import java.util.concurrent.Callable;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.evaluation.RetrievalEvaluation;

import lombok.RequiredArgsConstructor;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Spec;

/** CLI единой оценки ранее обученных моделей на подготовленной коллекции. */
@Component
@Command(
        name = "evaluate",
        mixinStandardHelpOptions = true,
        description = "Оценить LSA и Word2Vec по Recall@5/MRR@10 без переобучения."
)
@RequiredArgsConstructor
public class EvaluateCommand implements Callable<Integer> {

    private final RetrievalEvaluation evaluation;

    private final ObjectMapper jsonMapper;

    @Spec
    private CommandSpec specification;

    @Option(names = "--collection", required = true, description = "Каталог результата prepare-ria.")
    private Path collection;

    @Option(names = "--lsa-model", required = true, description = "Сохранённая модель LSA.")
    private Path lsaModel;

    @Option(names = "--word2vec-model", required = true, description = "Сохранённая модель Word2Vec.")
    private Path word2VecModel;

    @Option(names = "--output", required = true, description = "Новый каталог для report.json.")
    private Path output;

    /** Выводит краткую сводку; полный протокол сохраняется в output/report.json. */
    @Override
    public Integer call() throws Exception {
        var result = evaluation.evaluate(collection, lsaModel, word2VecModel, output);
        specification.commandLine().getOut()
                .println(jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

        return 0;
    }
}
