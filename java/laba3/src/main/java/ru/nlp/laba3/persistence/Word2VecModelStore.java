package ru.nlp.laba3.persistence;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.exception.ModelExportException;
import ru.nlp.laba3.exception.ModelImportException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.storage.ModelBundle;
import ru.nlp.laba3.model.word2vec.Word2VecManifest;
import ru.nlp.laba3.model.word2vec.Word2VecModel;
import ru.nlp.laba3.model.word2vec.Word2VecModelData;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.runtime.NativeRuntime;

/** Формат embeddings, манифест и проверки состояния Word2Vec. */
@Component
public class Word2VecModelStore extends AbstractModelStore<Word2VecModel, Word2VecManifest> {

    private static final String VECTORS_FILE = "vectors.txt";

    private static final String DATA_FILE = "data.json.gz";

    /** Связывает формат Word2Vec с общим жизненным циклом хранилища. */
    public Word2VecModelStore(ObjectMapper jsonMapper, TextPreprocessor preprocessor) {
        super(Algorithm.word2vec, jsonMapper, preprocessor, Word2VecManifest.class);
    }

    /** Сохраняет embeddings Word2Vec и сжатые документы/IDF. */
    @Override
    protected Map<String, String> writePayload(Path directory, ModelBundle<Word2VecModel> bundle) throws IOException {
        if (bundle.tokens() == null || bundle.zeroDocuments() == null) {
            throw new ModelExportException("Отсутствуют счётчики комплекта Word2Vec: " + directory);
        }
        String weightsHash = NativeRuntime.withThreads(bundle.model().parameters().workers(), () -> {
            try (var stream = Files.newOutputStream(directory.resolve(VECTORS_FILE))) {
                WordVectorSerializer.writeWordVectors(bundle.model().vectors().getLookupTable(), stream);
            }

            return FileChecksums.hash(directory.resolve(VECTORS_FILE));
        });
        var dataPath = directory.resolve(DATA_FILE);
        writeCompressed(dataPath, new Word2VecModelData(bundle.corpus(), bundle.model().idf()));

        return Map.of(VECTORS_FILE, weightsHash, DATA_FILE, FileChecksums.hash(dataPath));
    }

    /** Собирает типизированный манифест Word2Vec. */
    @Override
    protected Word2VecManifest createManifest(
            ModelBundle<Word2VecModel> bundle,
            Map<String, String> hashes,
            Map<String, Double> timings
    ) {
        return Word2VecManifest.builder()
                .parameters(bundle.model().parameters())
                .normalization(bundle.normalization())
                .stopwordsSha256(bundle.stopwordsSha256())
                .selection("first-unique-nonempty-in-source-order")
                .documents(bundle.corpus().documents().size())
                .scanned(bundle.corpus().scanned())
                .empty(bundle.corpus().empty())
                .duplicates(bundle.corpus().duplicates())
                .versions(bundle.versions())
                .weightsSha256(hashes.get(VECTORS_FILE))
                .dataSha256(hashes.get(DATA_FILE))
                .tokens(bundle.tokens())
                .vocabularySize(NativeRuntime.withLock(() -> bundle.model().vectors().vocab().numWords()))
                .zeroDocuments(bundle.zeroDocuments())
                .trainingSettings(bundle.model().trainingSettings())
                .seconds(timings)
                .build();
    }

    /** Проверяет параметры и профиль Word2Vec до чтения embeddings. */
    @Override
    protected void validateManifest(Word2VecManifest manifest) {
        if (manifest == null || manifest.parameters() == null) {
            throw new ModelImportException("В манифесте Word2Vec отсутствуют параметры модели");
        }
        validateNormalization(manifest.normalization(), manifest.stopwordsSha256());
    }

    /** Проверяет файлы и восстанавливает обученные embeddings Word2Vec. */
    @Override
    protected ModelBundle<Word2VecModel> readPayload(Path directory, Word2VecManifest manifest) throws IOException {
        var weights = directory.resolve(VECTORS_FILE);
        verifyChecksum(weights, manifest.weightsSha256(), "веса");
        var dataPath = directory.resolve(DATA_FILE);
        verifyChecksum(dataPath, manifest.dataSha256(), "данные");
        var data = readCompressed(dataPath, Word2VecModelData.class);

        if (
            data == null || data.corpus() == null || data.idf() == null || data.corpus().documents() == null
                    || data.corpus().documents().size() != manifest.documents()
        ) {
            throw new ModelImportException("Манифест не соответствует данным Word2Vec: " + directory);
        }

        return NativeRuntime.withThreads(manifest.parameters().workers(), () -> {
            var vectors = WordVectorSerializer.readWord2VecModel(weights.toFile(), false);

            if (
                vectors.getLayerSize() != manifest.parameters().dimensions()
                        || vectors.vocab().numWords() != manifest.vocabularySize()
            ) {
                throw new ModelImportException("Манифест не соответствует весам Word2Vec: " + directory);
            }
            var model = new Word2VecModel(vectors, data.idf(), manifest.parameters(), manifest.trainingSettings());

            return new ModelBundle<>(
                    data.corpus(),
                    model,
                    manifest.normalization(),
                    manifest.stopwordsSha256(),
                    manifest.versions(),
                    manifest.tokens(),
                    manifest.zeroDocuments(),
                    manifest.seconds()
            );
        });
    }
}
