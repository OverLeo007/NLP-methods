package ru.nlp.laba3.model.lsa;

import java.util.Map;

import lombok.Builder;

/** Метаданные комплекта LSA без корпуса, словаря и матриц проекции. */
@Builder
public record LsaManifest(
        /** Идентификатор алгоритма сохранённого комплекта. */
        String algorithm,
        /** Формула взвешивания терминов. */
        String weighting,
        /** Реализация и способ усечения SVD. */
        String svd,
        /** Способ выбора документов корпуса. */
        String selection,
        /** SHA-256 сжатого файла данных LSA. */
        String dataSha256,
        /** Число отобранных документов. */
        int documents,
        /** Число сохранённых терминов словаря. */
        int vocabularySize,
        /** Фактическое число сохранённых SVD-компонент. */
        int dimensions,
        /** Времена стадий; отсутствуют в ранее сохранённых манифестах. */
        Map<String, Double> seconds
) {

    /** Защищает замеры нового комплекта, сохраняя отсутствие поля в прежних файлах. */
    public LsaManifest {
        seconds = seconds == null ? null : Map.copyOf(seconds);
    }
}
