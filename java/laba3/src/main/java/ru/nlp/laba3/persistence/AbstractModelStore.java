package ru.nlp.laba3.persistence;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import com.fasterxml.jackson.databind.ObjectMapper;

import ru.nlp.laba3.exception.ModelExportException;
import ru.nlp.laba3.exception.ModelImportException;
import ru.nlp.laba3.io.FileChecksums;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.storage.ModelBundle;
import ru.nlp.laba3.preprocessing.TextPreprocessor;
import ru.nlp.laba3.runtime.StageTimings;

import lombok.AccessLevel;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Общий жизненный цикл файлового комплекта; M — состояние, F — тип манифеста. */
@RequiredArgsConstructor(access = AccessLevel.PROTECTED)
@Slf4j
public abstract class AbstractModelStore<M, F> implements ModelStore<M> {

    private static final String MANIFEST_FILE = "manifest.json";

    protected final Algorithm algorithm;

    private final ObjectMapper jsonMapper;

    private final TextPreprocessor preprocessor;

    private final Class<F> manifestType;

    /** Создаёт новый комплект и сохраняет одинаковые границы замеров для всех алгоритмов. */
    @Override
    public final Map<String, Double> save(Path directory, ModelBundle<M> bundle) {
        try {
            Files.createDirectories(directory.toAbsolutePath().getParent());
            Files.createDirectory(directory);
            log.info("[{}] Сохранение модели: {}", algorithm.displayName(), directory);
            long started = System.nanoTime();
            var hashes = writePayload(directory, bundle);
            var timings = StageTimings.withSaving(bundle.seconds(), started);
            var manifest = createManifest(bundle, hashes, timings);
            jsonMapper.writerWithDefaultPrettyPrinter().writeValue(directory.resolve(MANIFEST_FILE).toFile(), manifest);
            log.info("[{}] Модель сохранена: {}", algorithm.displayName(), directory);

            return timings;
        } catch (ModelExportException exception) {
            throw exception;
        } catch (IOException | RuntimeException exception) {
            throw new ModelExportException(
                    "Не удалось сохранить модель " + algorithm.displayName() + " в "
                            + directory + ": " + exception.getMessage(),
                    exception
            );
        }
    }

    /** Читает типизированный манифест и восстанавливает проверенное состояние без индекса. */
    @Override
    public final ModelBundle<M> load(Path directory) {
        try {
            log.info("[{}] Загрузка модели: {}", algorithm.displayName(), directory);
            var manifest = jsonMapper.readValue(directory.resolve(MANIFEST_FILE).toFile(), manifestType);
            validateManifest(manifest);
            var bundle = readPayload(directory, manifest);
            log.info(
                    "[{}] Модель загружена: {}, документов={}",
                    algorithm.displayName(),
                    directory,
                    bundle.corpus().documents().size()
            );

            return bundle;
        } catch (ModelImportException exception) {
            throw exception;
        } catch (IOException | RuntimeException exception) {
            throw new ModelImportException(
                    "Не удалось загрузить модель " + algorithm.displayName() + " из "
                            + directory + ": " + exception.getMessage(),
                    exception
            );
        }
    }

    /** Записывает данные и возвращает SHA-256 по именам файлов комплекта. */
    protected abstract Map<String, String> writePayload(Path directory, ModelBundle<M> bundle) throws IOException;

    /** Собирает манифест из состояния, контрольных сумм и завершённых замеров. */
    protected abstract F createManifest(ModelBundle<M> bundle, Map<String, String> hashes, Map<String, Double> timings);

    /** Проверяет метаданные до чтения файлов состояния. */
    protected abstract void validateManifest(F manifest);

    /** Проверяет файлы и восстанавливает состояние выбранного алгоритма. */
    protected abstract ModelBundle<M> readPayload(Path directory, F manifest) throws IOException;

    /** Записывает данные в общий формат gzip JSON. */
    protected final void writeCompressed(Path path, Object data) throws IOException {
        try (var stream = new GZIPOutputStream(Files.newOutputStream(path))) {
            jsonMapper.writeValue(stream, data);
        }
    }

    /** Читает типизированные данные из gzip JSON. */
    protected final <D> D readCompressed(Path path, Class<D> type) throws IOException {
        try (var stream = new GZIPInputStream(Files.newInputStream(path))) {
            return jsonMapper.readValue(stream, type);
        }
    }

    /** Проверяет SHA-256 и включает алгоритм и файл в сообщение повреждения. */
    protected final void verifyChecksum(Path path, String expected, String contents) throws IOException {
        if (!FileChecksums.hash(path).equals(expected)) {
            throw new ModelImportException("Повреждены " + contents + " " + algorithm.displayName() + ": " + path);
        }
    }

    /** Проверяет общий профиль нормализации и встроенный стоп-список. */
    protected final void validateNormalization(String profile, String stopwordsHash) {
        if (!preprocessor.profile().equals(profile) || !preprocessor.resourceHash().equals(stopwordsHash)) {
            throw new ModelImportException("Несовместимый профиль нормализации " + algorithm.displayName());
        }
    }
}
