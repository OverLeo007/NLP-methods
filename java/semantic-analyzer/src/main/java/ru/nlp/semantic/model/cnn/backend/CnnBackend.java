package ru.nlp.semantic.model.cnn.backend;

import org.springframework.stereotype.Component;

import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.backend.AbstractModelBackend;
import ru.nlp.semantic.model.base.store.ModelStore;
import ru.nlp.semantic.model.base.training.ModelTrainer;
import ru.nlp.semantic.model.base.vectorization.TextVectorizer;
import ru.nlp.semantic.model.cnn.CnnModel;
import ru.nlp.semantic.model.cnn.dto.CnnParameters;
import ru.nlp.semantic.model.cnn.vectorization.CnnVectorizer;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.RuntimeMetadata;

/** Подключение CNN к общим стадиям подготовки, обучения, индексации и сохранения. */
@Component
public final class CnnBackend extends AbstractModelBackend<CnnParameters, CnnModel> {

    /** Связывает типизированные компоненты CNN с общим сценарием. */
    public CnnBackend(
            CorpusPreparation preparation,
            TextPreprocessor preprocessor,
            ModelTrainer<CnnParameters, CnnModel> trainer,
            ModelStore<CnnModel> store,
            RuntimeMetadata runtimeMetadata
    ) {
        super(Algorithm.cnn, preparation, preprocessor, trainer, store, runtimeMetadata);
    }

    /** Создаёт CNN-векторизатор без изменения поискового индекса. */
    @Override
    protected TextVectorizer createVectorizer(CnnModel model) {
        return new CnnVectorizer(model);
    }

    /** Возвращает настройки потоков. */
    @Override
    protected int workers(CnnParameters parameters) {
        return parameters.workers();
    }

    /** Возвращает параметры загруженного состояния. */
    @Override
    protected CnnParameters parameters(CnnModel model) {
        return model.parameters();
    }
}
