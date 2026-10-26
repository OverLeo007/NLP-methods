package ru.nlp.laba3.service;

import org.springframework.stereotype.Service;

import ru.nlp.laba3.corpus.CorpusPreparation;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.lsa.LsaModel;
import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.persistence.ModelStore;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.runtime.RuntimeMetadata;
import ru.nlp.laba3.training.ModelTrainer;
import ru.nlp.laba3.vectorization.LsaVectorizer;
import ru.nlp.laba3.vectorization.TextVectorizer;

/** Настройка общего сценария для LSA без дублирования стадий обучения. */
@Service
public class LsaService extends AbstractModelService<LsaParameters, LsaModel> {

    /** Связывает общую подготовку и типизированные компоненты LSA. */
    public LsaService(
            CorpusPreparation preparation,
            TextPreprocessor preprocessor,
            ModelTrainer<LsaParameters, LsaModel> trainer,
            ModelStore<LsaModel> store,
            RuntimeMetadata runtimeMetadata
    ) {
        super(Algorithm.lsa, preparation, preprocessor, trainer, store, runtimeMetadata);
    }

    /** Создаёт векторизатор из обученного состояния LSA. */
    @Override
    protected TextVectorizer createVectorizer(LsaModel model) {
        return new LsaVectorizer(model.vocabulary(), model.idf(), model.projection());
    }

    /** Выбирает число потоков из параметров LSA. */
    @Override
    protected int workers(LsaParameters parameters) {
        return parameters.workers();
    }

    /** Читает параметры из сохранённого состояния LSA. */
    @Override
    protected LsaParameters parameters(LsaModel model) {
        return model.parameters();
    }
}
