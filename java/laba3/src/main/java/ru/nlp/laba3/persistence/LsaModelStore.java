package ru.nlp.laba3.persistence;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;

import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.exception.ModelImportException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.lsa.LsaManifest;
import ru.nlp.laba3.model.lsa.LsaModel;
import ru.nlp.laba3.model.lsa.LsaModelData;
import ru.nlp.laba3.model.storage.ModelBundle;
import ru.nlp.laba3.preprocessing.TextPreprocessor;

/** Формат данных, манифест и проверки состояния LSA. */
@Component
public class LsaModelStore extends AbstractModelStore<LsaModel, LsaManifest> {

    private static final String DATA_FILE = "lsa.json.gz";

    /** Связывает формат LSA с общим жизненным циклом хранилища. */
    public LsaModelStore(ObjectMapper jsonMapper, TextPreprocessor preprocessor) {
        super(Algorithm.lsa, jsonMapper, preprocessor, LsaManifest.class);
    }

    /** Сохраняет проекцию, корпус и метаданные LSA в gzip JSON. */
    @Override
    protected Map<String, String> writePayload(Path directory, ModelBundle<LsaModel> bundle) throws IOException {
        var path = directory.resolve(DATA_FILE);
        writeCompressed(path, createData(bundle));

        return Map.of(DATA_FILE, FileChecksums.hash(path));
    }

    /** Собирает типизированный манифест LSA. */
    @Override
    protected LsaManifest createManifest(
            ModelBundle<LsaModel> bundle,
            Map<String, String> hashes,
            Map<String, Double> timings
    ) {
        return LsaManifest.builder()
                .algorithm(algorithm.name())
                .weighting("sublinearTf-smoothedIdf-l2")
                .svd("openblas-lapack-dgesvd-thin-truncated")
                .selection("first-unique-nonempty-in-source-order")
                .dataSha256(hashes.get(DATA_FILE))
                .documents(bundle.corpus().documents().size())
                .vocabularySize(bundle.model().vocabulary().size())
                .dimensions(bundle.model().singularValues().length)
                .seconds(timings)
                .build();
    }

    /** Проверяет принадлежность комплекта алгоритму LSA. */
    @Override
    protected void validateManifest(LsaManifest manifest) {
        if (manifest == null || !algorithm.name().equals(manifest.algorithm())) {
            throw new ModelImportException("Ожидался комплект " + algorithm.displayName());
        }
    }

    /** Проверяет данные и восстанавливает готовую проекцию LSA. */
    @Override
    protected ModelBundle<LsaModel> readPayload(Path directory, LsaManifest manifest) throws IOException {
        var path = directory.resolve(DATA_FILE);
        verifyChecksum(path, manifest.dataSha256(), "данные");
        var data = readCompressed(path, LsaModelData.class);
        validate(data);

        if (
            manifest.documents() != data.corpus().documents().size()
                    || manifest.vocabularySize() != data.vocabulary().size()
                    || manifest.dimensions() != data.singularValues().length
        ) {
            throw new ModelImportException("Манифест не соответствует данным LSA: " + directory);
        }
        var model = new LsaModel(
                data.vocabulary(),
                data.idf(),
                data.projection(),
                data.singularValues(),
                data.parameters()
        );

        return new ModelBundle<>(
                data.corpus(),
                model,
                data.normalization(),
                data.stopwordsSha256(),
                data.versions(),
                data.tokens(),
                data.zeroDocuments(),
                manifest.seconds() == null ? data.seconds() : manifest.seconds()
        );
    }

    private LsaModelData createData(ModelBundle<LsaModel> bundle) {
        var model = bundle.model();

        return new LsaModelData(
                bundle.corpus(),
                model.vocabulary(),
                model.idf(),
                model.projection(),
                model.singularValues(),
                model.parameters(),
                bundle.normalization(),
                bundle.stopwordsSha256(),
                bundle.versions(),
                bundle.seconds(),
                bundle.tokens(),
                bundle.zeroDocuments()
        );
    }

    private void validate(LsaModelData data) {
        if (
            data == null || data.corpus() == null || data.corpus().documents() == null
                    || data.corpus().documents().size() < 2 || data.parameters() == null
                    || data.vocabulary() == null || data.idf() == null
                    || data.projection() == null || data.singularValues() == null
        ) {
            throw new ModelImportException("Неполные данные LSA");
        }

        validateNormalization(data.normalization(), data.stopwordsSha256());
        int terms = data.vocabulary().size();
        int dimensions = data.singularValues().length;

        if (
            terms < 2 || data.vocabulary().stream().anyMatch(term -> term == null || term.isBlank())
                    || new HashSet<>(data.vocabulary()).size() != terms
                    || data.idf().length != terms || data.projection().length != terms || dimensions < 1
                    || dimensions > Math.min(
                            data.parameters().dimensions(),
                            Math.min(terms, data.corpus().documents().size()) - 1
                    )
                    || Arrays.stream(data.idf()).anyMatch(value -> !Double.isFinite(value) || value < 1)
                    || Arrays.stream(data.singularValues()).anyMatch(value -> !Double.isFinite(value) || value < 0)
        ) {
            throw new ModelImportException("Несогласованные размеры или значения LSA");
        }

        for (var row : data.projection()) {
            if (
                row == null || row.length != dimensions
                        || Arrays.stream(row).anyMatch(value -> !Double.isFinite(value))
            ) {
                throw new ModelImportException("Некорректная SVD-проекция LSA");
            }
        }
    }
}
