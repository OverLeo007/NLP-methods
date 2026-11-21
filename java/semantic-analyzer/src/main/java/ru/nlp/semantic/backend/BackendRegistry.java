package ru.nlp.semantic.backend;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.springframework.stereotype.Component;

import ru.nlp.semantic.model.analysis.Algorithm;

/** Однозначный выбор backend по алгоритму без неявной подстановки другой модели. */
@Component
public final class BackendRegistry {

    private final Map<Algorithm, ModelBackend<?>> backends;

    /** Проверяет полноту и отсутствие повторных регистраций при запуске приложения. */
    public BackendRegistry(List<ModelBackend<?>> backends) {
        var registered = new EnumMap<Algorithm, ModelBackend<?>>(Algorithm.class);

        for (var backend : backends) {
            if (registered.putIfAbsent(backend.algorithm(), backend) != null) {
                throw new IllegalArgumentException("Повторный backend: " + backend.algorithm());
            }
        }

        for (var algorithm : Algorithm.values()) {
            if (!registered.containsKey(algorithm)) {
                throw new IllegalArgumentException("Отсутствует backend: " + algorithm);
            }
        }

        this.backends = Map.copyOf(registered);
    }

    /** Возвращает backend для открытия модели; обучение сохраняет типизированный контракт параметров. */
    public ModelBackend<?> require(Algorithm algorithm) {
        var backend = backends.get(algorithm);

        if (backend == null) {
            throw new IllegalArgumentException("Неизвестный алгоритм: " + algorithm);
        }

        return backend;
    }
}
