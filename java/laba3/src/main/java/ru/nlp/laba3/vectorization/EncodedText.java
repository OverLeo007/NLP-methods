package ru.nlp.laba3.vectorization;

/** Результат векторизации: нормализованный вектор либо нули и покрытие словаря. */
public record EncodedText(
        /** Координаты; массив защищён копированием при создании и чтении. */
        double[] values,
        /** Число известных вхождений токенов. */
        int knownTokens
) {

    /** Отделяет координаты результата от рабочего массива векторизатора. */
    public EncodedText {
        values = values.clone();
    }

    /** Возвращает копию координат для безопасного использования вызывающим кодом. */
    @Override
    public double[] values() {
        return values.clone();
    }

    /** Проверяет отсутствие ненулевых координат. */
    public boolean isZero() {
        for (double value : values) {
            if (value != 0) {
                return false;
            }
        }

        return true;
    }
}
