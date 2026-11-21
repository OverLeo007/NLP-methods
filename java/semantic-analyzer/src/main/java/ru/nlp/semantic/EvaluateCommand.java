package ru.nlp.semantic;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.concurrent.Callable;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.semantic.evaluation.RetrievalEvaluation;
import ru.nlp.semantic.model.analysis.Algorithm;
import ru.nlp.semantic.model.evaluation.EvaluationTarget;

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
        description = "Оценить LSA, Word2Vec и при указании --cnn-model CNN по Recall@5/MRR@10."
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

    @Option(names = "--cnn-model", description = "Сохранённая модель CNN; без параметра оцениваются LSA и Word2Vec.")
    private Path cnnModel;

    @Option(names = "--output", required = true, description = "Новый каталог для report.json.")
    private Path output;

    /** Выводит краткую сводку; полный протокол сохраняется в output/report.json. */
    @Override
    public Integer call() throws Exception {
        var targets = new ArrayList<EvaluationTarget>();
        targets.add(new EvaluationTarget(Algorithm.lsa, lsaModel));
        targets.add(new EvaluationTarget(Algorithm.word2vec, word2VecModel));

        if (cnnModel != null) {
            targets.add(new EvaluationTarget(Algorithm.cnn, cnnModel));
        }
        var result = evaluation.evaluate(collection, targets, output);
        specification.commandLine().getOut()
                .println(jsonMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

        return 0;
    }
}
