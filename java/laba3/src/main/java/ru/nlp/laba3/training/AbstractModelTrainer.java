package ru.nlp.laba3.training;

import java.io.IOException;
import java.util.List;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.exception.ModelTrainingException;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.training.TrainingResult;
import ru.nlp.laba3.runtime.NativeRuntime;
import ru.nlp.laba3.runtime.StageTimings;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Общие границы нативного обучения, замеров и обработки ошибок анализаторов. */
@RequiredArgsConstructor(access = AccessLevel.PROTECTED)
@Slf4j
public abstract class AbstractModelTrainer<P, M> implements ModelTrainer<P, M> {

    protected final Algorithm algorithm;

    /** Обучает состояние под нативным монитором и возвращает единый результат. */
    @Override
    public final TrainingResult<M> train(List<List<String>> documents, P parameters) {
        try {
            return NativeRuntime.withThreads(workers(parameters), () -> {
                long started = System.nanoTime();
                log.info("[{}] Обучение модели начато: документов={}", algorithm.displayName(), documents.size());
                var model = fit(documents, parameters);
                var result = new TrainingResult<>(
                        model,
                        vocabularySize(model),
                        dimensions(model),
                        StageTimings.elapsed(started)
                );
                log.info(
                        "[{}] Обучение модели завершено: словарь={}, размерность={}, время={} с",
                        algorithm.displayName(),
                        result.vocabularySize(),
                        result.dimensions(),
                        result.seconds()
                );

                return result;
            });
        } catch (InvalidInputException | ModelTrainingException exception) {
            throw exception;
        } catch (IOException | RuntimeException exception) {
            throw new ModelTrainingException(
                    "Не удалось обучить модель " + algorithm.displayName() + ": "
                            + exception.getMessage(),
                    exception
            );
        }
    }

    /** Выполняет обучение конкретного алгоритма внутри нативного монитора. */
    protected abstract M fit(List<List<String>> documents, P parameters);

    /** Возвращает число потоков обучения. */
    protected abstract int workers(P parameters);

    /** Возвращает фактический размер словаря обученного состояния. */
    protected abstract int vocabularySize(M model);

    /** Возвращает фактическую размерность обученного состояния. */
    protected abstract int dimensions(M model);
}
