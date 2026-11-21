package ru.nlp.semantic.model.word2vec.backend;

import org.springframework.stereotype.Component;

import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.backend.AbstractModelBackend;
import ru.nlp.semantic.model.base.store.ModelStore;
import ru.nlp.semantic.model.base.training.ModelTrainer;
import ru.nlp.semantic.model.base.vectorization.TextVectorizer;
import ru.nlp.semantic.model.word2vec.Word2VecModel;
import ru.nlp.semantic.model.word2vec.dto.Word2VecParameters;
import ru.nlp.semantic.model.word2vec.vectorization.Word2VecVectorizer;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.RuntimeMetadata;

/** Настройка общего сценария для Word2Vec без дублирования стадий обучения. */
@Component
public final class Word2VecBackend extends AbstractModelBackend<Word2VecParameters, Word2VecModel> {

    /** Связывает общую подготовку и типизированные компоненты Word2Vec. */
    public Word2VecBackend(
            CorpusPreparation preparation,
            TextPreprocessor preprocessor,
            ModelTrainer<Word2VecParameters, Word2VecModel> trainer,
            ModelStore<Word2VecModel> store,
            RuntimeMetadata runtimeMetadata
    ) {
        super(Algorithm.word2vec, preparation, preprocessor, trainer, store, runtimeMetadata);
    }

    /** Создаёт векторизатор из обученного состояния Word2Vec. */
    @Override
    protected TextVectorizer createVectorizer(Word2VecModel model) {
        return new Word2VecVectorizer(model.vectors(), model.idf());
    }

    /** Выбирает число потоков из параметров Word2Vec. */
    @Override
    protected int workers(Word2VecParameters parameters) {
        return parameters.workers();
    }

    /** Читает параметры из сохранённого состояния Word2Vec. */
    @Override
    protected Word2VecParameters parameters(Word2VecModel model) {
        return model.parameters();
    }
}
