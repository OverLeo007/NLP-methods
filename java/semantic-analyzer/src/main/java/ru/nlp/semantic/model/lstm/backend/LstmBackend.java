package ru.nlp.semantic.model.lstm.backend;

import org.springframework.stereotype.Component;

import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.backend.AbstractModelBackend;
import ru.nlp.semantic.model.base.store.ModelStore;
import ru.nlp.semantic.model.base.training.ModelTrainer;
import ru.nlp.semantic.model.base.vectorization.TextVectorizer;
import ru.nlp.semantic.model.lstm.LstmModel;
import ru.nlp.semantic.model.lstm.dto.LstmParameters;
import ru.nlp.semantic.model.lstm.vectorization.LstmVectorizer;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.RuntimeMetadata;

/** Подключение LSTM к общим стадиям подготовки, обучения, индексации и сохранения. */
@Component
public final class LstmBackend extends AbstractModelBackend<LstmParameters, LstmModel> {

    /** Связывает типизированные компоненты LSTM с общим сценарием. */
    public LstmBackend(
            CorpusPreparation preparation,
            TextPreprocessor preprocessor,
            ModelTrainer<LstmParameters, LstmModel> trainer,
            ModelStore<LstmModel> store,
            RuntimeMetadata runtimeMetadata
    ) {
        super(Algorithm.lstm, preparation, preprocessor, trainer, store, runtimeMetadata);
    }

    /** Создаёт LSTM-векторизатор без изменения поискового индекса. */
    @Override
    protected TextVectorizer createVectorizer(LstmModel model) {
        return new LstmVectorizer(model);
    }

    /** Возвращает настройки потоков. */
    @Override
    protected int workers(LstmParameters parameters) {
        return parameters.workers();
    }

    /** Возвращает параметры загруженного состояния. */
    @Override
    protected LstmParameters parameters(LstmModel model) {
        return model.parameters();
    }
}
