package ru.nlp.laba3.service;

import org.springframework.stereotype.Service;

import ru.nlp.laba3.corpus.CorpusPreparation;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.word2vec.Word2VecModel;
import ru.nlp.laba3.model.word2vec.Word2VecParameters;
import ru.nlp.laba3.persistence.ModelStore;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.runtime.RuntimeMetadata;
import ru.nlp.laba3.training.ModelTrainer;
import ru.nlp.laba3.vectorization.TextVectorizer;
import ru.nlp.laba3.vectorization.Word2VecVectorizer;

/** Настройка общего сценария для Word2Vec без дублирования стадий обучения. */
@Service
public class Word2VecService extends AbstractModelService<Word2VecParameters, Word2VecModel> {

    /** Связывает общую подготовку и типизированные компоненты Word2Vec. */
    public Word2VecService(
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
