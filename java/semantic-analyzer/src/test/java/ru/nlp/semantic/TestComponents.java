package ru.nlp.semantic;

import java.io.IOException;
import java.util.List;

import ru.nlp.semantic.backend.Word2VecBackend;
import ru.nlp.semantic.config.JsonConfiguration;
import ru.nlp.semantic.config.PreprocessingConfiguration;
import ru.nlp.semantic.corpus.CorpusLoader;
import ru.nlp.semantic.corpus.CorpusPreparation;
import ru.nlp.semantic.corpus.reader.CompositeCorpusReader;
import ru.nlp.semantic.corpus.reader.CorpusReader;
import ru.nlp.semantic.corpus.reader.CsvCorpusReader;
import ru.nlp.semantic.corpus.reader.JsonlCorpusReader;
import ru.nlp.semantic.corpus.reader.TxtCorpusReader;
import ru.nlp.semantic.persistence.Word2VecModelStore;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.RuntimeMetadata;
import ru.nlp.semantic.training.Word2VecTrainer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Компоненты узких тестов с той же конфигурацией мапперов и нормализации, что у CLI. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class TestComponents {

    /** Создаёт общую предобработку с production-композицией цепочки. */
    public static TextPreprocessor preprocessor() throws IOException {
        return new TextPreprocessor(new PreprocessingConfiguration().textNormalizationChain());
    }

    /** Создаёт отбор корпуса поверх тех же форматов, что доступны CLI. */
    public static CorpusLoader loader() throws IOException {
        return new CorpusLoader(corpusReader(), preprocessor());
    }

    /** Создаёт потоковое чтение через общий контракт и производственные форматы. */
    public static CorpusReader corpusReader() {
        var configuration = new JsonConfiguration();

        return new CompositeCorpusReader(
                List.of(
                        new JsonlCorpusReader(configuration.jsonMapper()),
                        new CsvCorpusReader(configuration.csvMapper()),
                        new TxtCorpusReader()
                )
        );
    }

    /** Создаёт backend с общей подготовкой и одним JSON mapper для чтения и хранения. */
    public static Word2VecBackend word2VecBackend() throws IOException {
        var configuration = new JsonConfiguration();
        var mapper = configuration.jsonMapper();
        var preprocessor = preprocessor();
        var reader = new CorpusLoader(corpusReader(), preprocessor);

        return new Word2VecBackend(
                new CorpusPreparation(reader, preprocessor),
                preprocessor,
                new Word2VecTrainer(),
                new Word2VecModelStore(mapper, preprocessor),
                new RuntimeMetadata(mapper)
        );
    }
}
