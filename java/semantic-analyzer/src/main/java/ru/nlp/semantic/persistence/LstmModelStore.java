package ru.nlp.semantic.persistence;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

import org.deeplearning4j.util.ModelSerializer;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.semantic.exception.ModelExportException;
import ru.nlp.semantic.exception.ModelImportException;
import ru.nlp.semantic.io.FileChecksums;
import ru.nlp.semantic.model.analysis.Algorithm;
import ru.nlp.semantic.model.lstm.LstmManifest;
import ru.nlp.semantic.model.lstm.LstmModel;
import ru.nlp.semantic.model.lstm.LstmModelData;
import ru.nlp.semantic.model.storage.ModelBundle;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.NativeRuntime;
import ru.nlp.semantic.vectorization.SequenceVocabulary;

/** Хранилище LSTM: DL4J ZIP без updater, сжатый корпус/словарь и проверяемый манифест. */
@Component
public final class LstmModelStore extends AbstractModelStore<LstmModel, LstmManifest> {

    private static final String WEIGHTS_FILE = "network.zip";

    private static final String DATA_FILE = "data.json.gz";

    /** Подключает LSTM к общему жизненному циклу файлового комплекта. */
    public LstmModelStore(ObjectMapper jsonMapper, TextPreprocessor preprocessor) {
        super(Algorithm.lstm, jsonMapper, preprocessor, LstmManifest.class);
    }

    /** Записывает веса для inference и независимый от Java-пакетов JSON данных. */
    @Override
    protected Map<String, String> writePayload(Path directory, ModelBundle<LstmModel> bundle) throws IOException {
        if (bundle.tokens() == null || bundle.zeroDocuments() == null) {
            throw new ModelExportException("LSTM: отсутствуют счётчики комплекта");
        }
        NativeRuntime.withThreads(bundle.model().parameters().workers(), () -> {
            ModelSerializer.writeModel(bundle.model().network(), directory.resolve(WEIGHTS_FILE).toFile(), false);

            return null;
        });
        writeCompressed(directory.resolve(DATA_FILE), new LstmModelData(bundle.corpus(), bundle.model().vocabulary()));

        return Map.of(
                WEIGHTS_FILE,
                FileChecksums.hash(directory.resolve(WEIGHTS_FILE)),
                DATA_FILE,
                FileChecksums.hash(directory.resolve(DATA_FILE))
        );
    }

    /** Сохраняет сведения обучения без добавления LSTM-полей в общую сводку. */
    @Override
    protected LstmManifest createManifest(
            ModelBundle<LstmModel> bundle,
            Map<String, String> hashes,
            Map<String, Double> timings
    ) {
        return LstmManifest.builder()
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
                .initialTokenLoss(bundle.model().initialTokenLoss())
                .finalTokenLoss(bundle.model().finalTokenLoss())
                .weightsSha256(hashes.get(WEIGHTS_FILE))
                .dataSha256(hashes.get(DATA_FILE))
                .build();
    }

    /** Проверяет версию формата, общую предобработку и обязательные сведения. */
    @Override
    protected void validateManifest(LstmManifest manifest) {
        if (
            manifest == null || !algorithm.name().equals(manifest.algorithm()) || manifest.formatVersion() != 1
                    || manifest.parameters() == null || !Double.isFinite(manifest.initialTokenLoss())
                    || !Double.isFinite(manifest.finalTokenLoss()) || manifest.versions() == null
                    || manifest.seconds() == null || manifest.documents() < 2 || manifest.vocabularySize() < 2
                    || manifest.vocabularySize() > manifest.parameters().maxFeatures() || manifest.tokens() < 2
                    || manifest.zeroDocuments() < 0 || manifest.zeroDocuments() > manifest.documents()

        ) {
            throw new ModelImportException("LSTM: неподдерживаемый или неполный манифест");
        }
        validateNormalization(manifest.normalization(), manifest.stopwordsSha256());
    }

    /** Проверяет обе контрольные суммы и восстанавливает кодировщик без обучения. */
    @Override
    protected ModelBundle<LstmModel> readPayload(Path directory, LstmManifest manifest) throws IOException {
        var weights = directory.resolve(WEIGHTS_FILE);
        var dataPath = directory.resolve(DATA_FILE);
        verifyChecksum(weights, manifest.weightsSha256(), "веса");
        verifyChecksum(dataPath, manifest.dataSha256(), "корпус и словарь");
        var data = readCompressed(dataPath, LstmModelData.class);

        if (
            data == null || data.corpus() == null || data.corpus().documents() == null || data.vocabulary() == null
                    || data.corpus().documents().size() != manifest.documents()
                    || data.vocabulary().size() != manifest.vocabularySize()
        ) {
            throw new ModelImportException("LSTM: данные не соответствуют манифесту");
        }
        var vocabulary = new SequenceVocabulary(data.vocabulary());

        return NativeRuntime.withThreads(manifest.parameters().workers(), () -> {
            var network = ModelSerializer.restoreComputationGraph(weights.toFile(), false);

            if (
                network.getLayer("encoder").getParam("RW").size(0) != manifest.parameters().dimensions()
                        || network.getLayer("embedding").getParam("W").size(0) != vocabulary.words().size() + 2
                        || network.getLayer("embedding").getParam("W").size(1) != manifest.parameters()
                                .embeddingDimensions()
                        || network.getLayer("words").getParam("b").length() != vocabulary.words().size()
                        || !Double.isFinite(network.params().norm2Number().doubleValue())
            ) {
                throw new ModelImportException("LSTM: веса сети не соответствуют манифесту");
            }
            var model = new LstmModel(
                    network,
                    vocabulary.words(),
                    manifest.parameters(),
                    manifest.initialTokenLoss(),
                    manifest.finalTokenLoss()
            );

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
