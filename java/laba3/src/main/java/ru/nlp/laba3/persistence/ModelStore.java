package ru.nlp.laba3.persistence;

import java.nio.file.Path;
import java.util.Map;

import ru.nlp.laba3.model.storage.ModelBundle;

/** Хранение состояния M и доступных метаданных без обучения и построения индекса. */
public interface ModelStore<M> {

    /** Создаёт новый каталог и возвращает сохранённые времена стадий, включая запись данных до манифеста. */
    Map<String, Double> save(Path directory, ModelBundle<M> bundle);

    /** Проверяет целостность и возвращает состояние модели вместе с корпусом. */
    ModelBundle<M> load(Path directory);
}
