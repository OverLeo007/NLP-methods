package ru.nlp.semantic.model.base.training;

import java.util.List;

import ru.nlp.semantic.model.base.dto.TrainingResult;

/** Типизированное обучение состояния модели M с параметрами P только по документам. */
public interface ModelTrainer<P, M> {

    /** Обучает преобразование; самостоятельно защищает используемые нативные операции. */
    TrainingResult<M> train(List<List<String>> documents, P parameters);
}
