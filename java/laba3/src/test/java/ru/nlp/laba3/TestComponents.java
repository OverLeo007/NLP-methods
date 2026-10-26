package ru.nlp.laba3;

import java.io.IOException;
import java.util.List;

import ru.nlp.laba3.backend.Word2VecBackend;
import ru.nlp.laba3.config.JsonConfiguration;
import ru.nlp.laba3.config.PreprocessingConfiguration;
import ru.nlp.laba3.corpus.CorpusLoader;
import ru.nlp.laba3.corpus.CorpusPreparation;
import ru.nlp.laba3.corpus.reader.CompositeCorpusReader;
import ru.nlp.laba3.corpus.reader.CorpusReader;
import ru.nlp.laba3.corpus.reader.CsvCorpusReader;
import ru.nlp.laba3.corpus.reader.JsonlCorpusReader;
import ru.nlp.laba3.corpus.reader.TxtCorpusReader;
import ru.nlp.laba3.persistence.Word2VecModelStore;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.runtime.RuntimeMetadata;
import ru.nlp.laba3.service.Word2VecService;
import ru.nlp.laba3.training.Word2VecTrainer;

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

    /** Создаёт backend с production-сервисом и одним JSON mapper для чтения и хранения. */
    public static Word2VecBackend word2VecBackend() throws IOException {
        var configuration = new JsonConfiguration();
        var mapper = configuration.jsonMapper();
        var preprocessor = preprocessor();
        var reader = new CorpusLoader(corpusReader(), preprocessor);

        return new Word2VecBackend(
                new Word2VecService(
                        new CorpusPreparation(reader, preprocessor),
                        preprocessor,
                        new Word2VecTrainer(),
                        new Word2VecModelStore(mapper, preprocessor),
                        new RuntimeMetadata(mapper)
                )
        );
    }
}
