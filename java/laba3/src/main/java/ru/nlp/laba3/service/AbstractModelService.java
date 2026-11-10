package ru.nlp.laba3.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import ru.nlp.laba3.corpus.CorpusPreparation;
import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.corpus.Corpus;
import ru.nlp.laba3.model.corpus.PreparedCorpus;
import ru.nlp.laba3.model.search.OpenedModel;
import ru.nlp.laba3.model.storage.ModelBundle;
import ru.nlp.laba3.model.storage.ModelMetadata;
import ru.nlp.laba3.model.training.TrainingResult;
import ru.nlp.laba3.model.training.TrainingSummary;
import ru.nlp.laba3.persistence.ModelStore;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.runtime.NativeRuntime;
import ru.nlp.laba3.runtime.RuntimeMetadata;
import ru.nlp.laba3.runtime.StageTimings;
import ru.nlp.laba3.search.SearchSession;
import ru.nlp.laba3.training.ModelTrainer;
import ru.nlp.laba3.vectorization.TextVectorizer;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Общий сценарий анализатора; алгоритм определяет только векторизатор и потоки. */
@RequiredArgsConstructor(access = AccessLevel.PROTECTED)
@Slf4j
public abstract class AbstractModelService<P, M> {

    private final Algorithm algorithm;

    private final CorpusPreparation preparation;

    private final TextPreprocessor preprocessor;

    private final ModelTrainer<P, M> trainer;

    private final ModelStore<M> store;

    private final RuntimeMetadata runtimeMetadata;

    /** Выполняет одинаковые стадии обучения и сохраняет полный комплект без перезаписи. */
    public final TrainingSummary train(Path input, Path output, int limit, P parameters) throws IOException {
        checkOutput(output);
        log.info("[{}] Стадия: подготовка корпуса", algorithm.displayName());
        long started = System.nanoTime();
        var prepared = preparation.prepare(input, limit);
        double preparationSeconds = StageTimings.elapsed(started);

        log.info("[{}] Стадия: обучение", algorithm.displayName());
        started = System.nanoTime();
        var trained = trainer.train(prepared.documents(), parameters);
        double trainingSeconds = StageTimings.elapsed(started);

        log.info("[{}] Стадия: построение индекса", algorithm.displayName());
        started = System.nanoTime();
        var session = createSession(prepared.corpus(), trained.model());
        double indexingSeconds = StageTimings.elapsed(started);

        if (session.zeroDocuments() == prepared.corpus().documents().size()) {
            throw new InvalidInputException("Все документы " + algorithm.displayName() + " имеют нулевые векторы");
        }

        var timings = Map
                .of("preparation", preparationSeconds, "training", trainingSeconds, "indexing", indexingSeconds);
        var bundle = createBundle(prepared, trained, session.zeroDocuments(), parameters, timings);
        log.info("[{}] Стадия: сохранение модели в {}", algorithm.displayName(), output);
        var completedTimings = store.save(output, bundle);
        log.info(
                "[{}] Сценарий обучения завершён: {}, времена стадий, с={}",
                algorithm.displayName(),
                output,
                completedTimings
        );

        return createSummary(prepared, trained, session.zeroDocuments(), output, completedTimings);
    }

    /** Открывает готовый комплект для повторных запросов без обучения. */
    public final SearchSession open(Path directory) {
        return openModel(directory).session();
    }

    /** Открывает то же состояние, сохраняя общие метаданные для проверок и оценки. */
    public final OpenedModel<P> openModel(Path directory) {
        var bundle = store.load(directory);

        return new OpenedModel<>(
                createSession(bundle.corpus(), bundle.model()),
                new ModelMetadata<>(
                        parameters(bundle.model()),
                        bundle.normalization(),
                        bundle.stopwordsSha256(),
                        bundle.versions(),
                        bundle.seconds(),
                        bundle.tokens()
                )
        );
    }

    /** Возвращает типизированные параметры из загруженного состояния алгоритма. */
    protected abstract P parameters(M model);

    /** Создаёт векторизатор обученного состояния; он сам защищает нативное чтение. */
    protected abstract TextVectorizer createVectorizer(M model);

    /** Возвращает число потоков для согласованного снимка параметров нативной среды. */
    protected abstract int workers(P parameters);

    private void checkOutput(Path output) {
        if (Files.exists(output)) {
            throw new InvalidInputException("Каталог модели " + algorithm.displayName() + " уже существует: " + output);
        }
    }

    private SearchSession createSession(Corpus corpus, M model) {
        return SearchSession.create(corpus.documents(), preprocessor, createVectorizer(model));
    }

    private ModelBundle<M> createBundle(
            PreparedCorpus prepared,
            TrainingResult<M> trained,
            long zeros,
            P parameters,
            Map<String, Double> timings
    ) throws IOException {
        var versions = NativeRuntime.withThreads(workers(parameters), runtimeMetadata::versions);

        return new ModelBundle<>(
                prepared.corpus(),
                trained.model(),
                preprocessor.profile(),
                preprocessor.resourceHash(),
                versions,
                prepared.tokens(),
                zeros,
                timings
        );
    }

    private TrainingSummary createSummary(
            PreparedCorpus prepared,
            TrainingResult<M> trained,
            long zeros,
            Path output,
            Map<String, Double> timings
    ) {
        return new TrainingSummary(
                prepared.corpus().documents().size(),
                prepared.tokens(),
                trained.vocabularySize(),
                trained.dimensions(),
                zeros,
                timings.get("training"),
                output.toString(),
                timings
        );
    }
}
