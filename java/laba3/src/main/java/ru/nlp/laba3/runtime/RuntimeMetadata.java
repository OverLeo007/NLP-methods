package ru.nlp.laba3.runtime;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Properties;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.RequiredArgsConstructor;

/** Версии библиотек из ресурсов сборки и фактические параметры окружения. */
@Component
@RequiredArgsConstructor
public class RuntimeMetadata {

    private final ObjectMapper jsonMapper;

    /** Собирает воспроизводимые сведения об окружении для манифеста. */
    public Map<String, String> versions() throws IOException {
        var properties = new Properties();

        try (var stream = RuntimeMetadata.class.getResourceAsStream("/model-versions.properties")) {

            if (stream == null) {
                throw new IOException("Отсутствует model-versions.properties");
            }

            properties.load(stream);
        }

        var values = new LinkedHashMap<String, String>();
        properties.stringPropertyNames().stream().sorted()
                .forEach(key -> values.put(key, properties.getProperty(key)));
        values.put("java", System.getProperty("java.version"));
        values.put("os", System.getProperty("os.name"));
        values.put("arch", System.getProperty("os.arch"));
        values.put("nd4jBackend", Nd4j.getBackend().getClass().getName());
        values.put(
                "nd4jOmpMaxThreads",
                Integer.toString(
                        NativeOpsHolder.getInstance().getDeviceNativeOps().ompGetMaxThreads()
                )
        );
        values.put("jackson", jsonMapper.version().toFullString());

        // Сохраняем порядок отсортированных версий и следующих за ними сведений среды.

        return Collections.unmodifiableMap(values);
    }
}
