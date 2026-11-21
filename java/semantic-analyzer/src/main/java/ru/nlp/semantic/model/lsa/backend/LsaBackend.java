package ru.nlp.semantic.model.lsa.backend;

import org.springframework.stereotype.Component;

import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.backend.AbstractModelBackend;
import ru.nlp.semantic.model.base.store.ModelStore;
import ru.nlp.semantic.model.base.training.ModelTrainer;
import ru.nlp.semantic.model.base.vectorization.TextVectorizer;
import ru.nlp.semantic.model.lsa.LsaModel;
import ru.nlp.semantic.model.lsa.dto.LsaParameters;
import ru.nlp.semantic.model.lsa.vectorization.LsaVectorizer;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.RuntimeMetadata;

/** Настройка общего сценария для LSA без дублирования стадий обучения. */
@Component
public final class LsaBackend extends AbstractModelBackend<LsaParameters, LsaModel> {

    /** Связывает общую подготовку и типизированные компоненты LSA. */
    public LsaBackend(
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
