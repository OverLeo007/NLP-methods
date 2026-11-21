package ru.nlp.semantic.model.cnn.store;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import org.deeplearning4j.util.ModelSerializer;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.semantic.exception.ModelExportException;
import ru.nlp.semantic.exception.ModelImportException;
import ru.nlp.semantic.io.FileChecksums;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.dto.ModelBundle;
import ru.nlp.semantic.model.base.store.AbstractModelStore;
import ru.nlp.semantic.model.cnn.CnnModel;
import ru.nlp.semantic.model.cnn.dto.CnnManifest;
import ru.nlp.semantic.model.cnn.dto.CnnModelData;
import ru.nlp.semantic.model.cnn.network.CnnNetwork;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.NativeRuntime;

/** Хранилище CNN: DL4J ZIP без updater, сжатый корпус/словарь и проверяемый манифест. */
@Component
public final class CnnModelStore extends AbstractModelStore<CnnModel, CnnManifest> {

    private static final String WEIGHTS_FILE = "network.zip";

    private static final String DATA_FILE = "data.json.gz";

    /** Подключает CNN к общему жизненному циклу файлового комплекта. */
    public CnnModelStore(ObjectMapper jsonMapper, TextPreprocessor preprocessor) {
        super(Algorithm.cnn, jsonMapper, preprocessor, CnnManifest.class);
    }

    /** Записывает веса для inference и независимый от Java-пакетов JSON данных. */
    @Override
    protected Map<String, String> writePayload(Path directory, ModelBundle<CnnModel> bundle) throws IOException {
        if (bundle.tokens() == null || bundle.zeroDocuments() == null) {
            throw new ModelExportException("CNN: отсутствуют счётчики комплекта");
        }
        NativeRuntime.withThreads(bundle.model().parameters().workers(), () -> {
            ModelSerializer.writeModel(bundle.model().network(), directory.resolve(WEIGHTS_FILE).toFile(), false);

            return null;
        });
        writeCompressed(directory.resolve(DATA_FILE), new CnnModelData(bundle.corpus(), bundle.model().vocabulary()));

        return Map.of(
                WEIGHTS_FILE,
                FileChecksums.hash(directory.resolve(WEIGHTS_FILE)),
                DATA_FILE,
                FileChecksums.hash(directory.resolve(DATA_FILE))
        );
    }

    /** Сохраняет сведения обучения без добавления CNN-полей в общую сводку. */
    @Override
    protected CnnManifest createManifest(
            ModelBundle<CnnModel> bundle,
            Map<String, String> hashes,
            Map<String, Double> timings
    ) {
        return CnnManifest.builder()
                .algorithm(algorithm.name())
                .formatVersion(1)
                .parameters(bundle.model().parameters())
                .normalization(bundle.normalization())
                .stopwordsSha256(bundle.stopwordsSha256())
                .versions(bundle.versions())
                .seconds(timings)
                .tokens(bundle.tokens())
                .zeroDocuments(bundle.zeroDocuments())
                .documents(bundle.corpus().documents().size())
                .vocabularySize(bundle.model().vocabulary().size())
                .training(bundle.model().training())
                .weightsSha256(hashes.get(WEIGHTS_FILE))
                .dataSha256(hashes.get(DATA_FILE))
                .build();
    }

    /** Проверяет версию формата, общую предобработку и обязательные сведения. */
    @Override
    protected void validateManifest(CnnManifest manifest) {
        if (
            manifest == null || !algorithm.name().equals(manifest.algorithm()) || manifest.formatVersion() != 1
                    || manifest.parameters() == null || manifest.training() == null || manifest.versions() == null
                    || manifest.seconds() == null || manifest.documents() < 2 || manifest.vocabularySize() < 2
                    || manifest.vocabularySize() > manifest.parameters().maxFeatures() || manifest.tokens() < 2
                    || manifest.zeroDocuments() < 0 || manifest.zeroDocuments() > manifest.documents()
                    || manifest.training().trainingDocuments() > manifest.documents()
                    || manifest.training().trainingDocuments() + manifest.training().validationDocuments() > manifest
                            .documents()
                    || ((manifest.parameters().validationFraction() == 0) != (manifest.training()
                            .validationDocuments() == 0))
                    || manifest.training().epochLosses().size() != manifest.parameters().epochs()
        ) {
            throw new ModelImportException("CNN: неподдерживаемый или неполный манифест");
        }
        validateNormalization(manifest.normalization(), manifest.stopwordsSha256());
    }

    /** Проверяет обе контрольные суммы и восстанавливает кодировщик без обучения. */
    @Override
    protected ModelBundle<CnnModel> readPayload(Path directory, CnnManifest manifest) throws IOException {
        var weights = directory.resolve(WEIGHTS_FILE);
        var dataPath = directory.resolve(DATA_FILE);
        verifyChecksum(weights, manifest.weightsSha256(), "веса");
        verifyChecksum(dataPath, manifest.dataSha256(), "корпус и словарь");
        var data = readCompressed(dataPath, CnnModelData.class);

        if (
            data == null || data.corpus() == null || data.corpus().documents() == null || data.vocabulary() == null
                    || data.corpus().documents().size() != manifest.documents()
                    || data.vocabulary().size() != manifest.vocabularySize()
        ) {
            throw new ModelImportException("CNN: данные не соответствуют манифесту");
        }
        var vocabulary = new SequenceVocabulary(data.vocabulary());

        return NativeRuntime.withThreads(manifest.parameters().workers(), () -> {
            var network = ModelSerializer.restoreMultiLayerNetwork(weights.toFile(), false);
            CnnNetwork.validate(network, manifest.parameters(), vocabulary.words().size());
            var model = new CnnModel(network, vocabulary.words(), manifest.parameters(), manifest.training());

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
