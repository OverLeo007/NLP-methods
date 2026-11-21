package ru.nlp.semantic.model.lstm.store;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.conf.graph.rnn.LastTimeStepVertex;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.util.ModelSerializer;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.semantic.exception.ModelExportException;
import ru.nlp.semantic.exception.ModelImportException;
import ru.nlp.semantic.io.FileChecksums;
import ru.nlp.semantic.model.Algorithm;
import ru.nlp.semantic.model.base.dto.ModelBundle;
import ru.nlp.semantic.model.base.store.AbstractModelStore;
import ru.nlp.semantic.model.lstm.LstmModel;
import ru.nlp.semantic.model.lstm.dto.LstmManifest;
import ru.nlp.semantic.model.lstm.dto.LstmModelData;
import ru.nlp.semantic.model.lstm.dto.LstmObjective;
import ru.nlp.semantic.model.shared.vectorization.SequenceVocabulary;
import ru.nlp.semantic.preprocessing.TextPreprocessor;
import ru.nlp.semantic.runtime.NativeRuntime;

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
        boolean seq2seq = bundle.model().parameters().objective() == LstmObjective.seq2seq;

        return LstmManifest.builder()
                .algorithm(algorithm.name())
                .formatVersion(seq2seq ? 1 : 2)
                .parameters(bundle.model().parameters())
                .normalization(bundle.normalization())
                .stopwordsSha256(bundle.stopwordsSha256())
                .versions(bundle.versions())
                .seconds(timings)
                .tokens(bundle.tokens())
                .zeroDocuments(bundle.zeroDocuments())
                .documents(bundle.corpus().documents().size())
                .vocabularySize(bundle.model().vocabulary().size())
                .initialTokenLoss(seq2seq ? bundle.model().initialLoss() : null)
                .finalTokenLoss(seq2seq ? bundle.model().finalLoss() : null)
                .initialPairLoss(seq2seq ? null : bundle.model().initialLoss())
                .finalPairLoss(seq2seq ? null : bundle.model().finalLoss())
                .weightsSha256(hashes.get(WEIGHTS_FILE))
                .dataSha256(hashes.get(DATA_FILE))
                .build();
    }

    /** Проверяет версию формата, общую предобработку и обязательные сведения. */
    @Override
    protected void validateManifest(LstmManifest manifest) {
        if (
            manifest == null || !algorithm.name().equals(manifest.algorithm())
                    || manifest.parameters() == null || manifest.versions() == null
                    || manifest.seconds() == null || manifest.documents() < 2 || manifest.vocabularySize() < 2
                    || manifest.vocabularySize() > manifest.parameters().maxFeatures() || manifest.tokens() < 2
                    || manifest.zeroDocuments() < 0 || manifest.zeroDocuments() > manifest.documents()
        ) {
            throw new ModelImportException("LSTM: неподдерживаемый или неполный манифест");
        }
        boolean seq2seq = manifest.parameters().objective() == LstmObjective.seq2seq;
        Double initial = seq2seq ? manifest.initialTokenLoss() : manifest.initialPairLoss();
        Double last = seq2seq ? manifest.finalTokenLoss() : manifest.finalPairLoss();

        if (
            manifest.formatVersion() != (seq2seq ? 1 : 2) || initial == null || last == null
                    || !Double.isFinite(initial) || !Double.isFinite(last)
        ) {
            throw new ModelImportException("LSTM: версия или диагностика не соответствует задаче обучения");
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
            boolean seq2seq = manifest.parameters().objective() == LstmObjective.seq2seq;
            boolean matchingHead = seq2seq
                    ? network.getConfiguration().getVertices().get("context") instanceof LastTimeStepVertex
                            && network.getConfiguration().getNetworkOutputs().equals(List.of("words"))
                            && network.getLayer("words").getParam("b").length() == vocabulary.words().size()
                    : network.getConfiguration().getNetworkOutputs().equals(List.of("pairs"))
                            && network.getLayer("context").conf().getLayer() instanceof GlobalPoolingLayer pooling
                            && pooling.getPoolingType() == PoolingType.AVG;

            if (
                network.getLayer("encoder").getParam("RW").size(0) != manifest.parameters().dimensions()
                        || network.getLayer("embedding").getParam("W").size(0) != vocabulary.words().size() + 2
                        || network.getLayer("embedding").getParam("W").size(1) != manifest.parameters()
                                .embeddingDimensions()
                        || !matchingHead
                        || !Double.isFinite(network.params().norm2Number().doubleValue())
            ) {
                throw new ModelImportException("LSTM: веса сети не соответствуют манифесту");
            }
            var model = new LstmModel(
                    network,
                    vocabulary.words(),
                    manifest.parameters(),
                    seq2seq ? manifest.initialTokenLoss() : manifest.initialPairLoss(),
                    seq2seq ? manifest.finalTokenLoss() : manifest.finalPairLoss()
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
