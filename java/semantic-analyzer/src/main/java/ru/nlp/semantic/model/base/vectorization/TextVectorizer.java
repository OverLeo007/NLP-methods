package ru.nlp.semantic.model.base.vectorization;

import java.util.List;

import ru.nlp.semantic.model.base.dto.EncodedText;

/** Преобразование уже нормализованных токенов без знания о ранжировании и файлах. */
public interface TextVectorizer {

    /** Возвращает фиксированное число координат каждого результата. */
    int dimensions();

    /** Возвращает конечный L2-нормализованный вектор либо нули и число известных токенов. */
    EncodedText encode(List<String> tokens);
}
