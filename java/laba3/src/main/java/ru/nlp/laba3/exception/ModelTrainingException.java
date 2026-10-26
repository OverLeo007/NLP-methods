package ru.nlp.laba3.exception;

/** Неожиданная ошибка библиотеки обучения или нативной среды. Причина и трассировка сохраняются. */
public final class ModelTrainingException extends NlpException {

    /** Создаёт ошибку обучения с диагностической трассировкой. */
    public ModelTrainingException(String message) {
        this(message, null);
    }

    /** Сохраняет исходную ошибку для диагностики сбоя операции. */
    public ModelTrainingException(String message, Throwable cause) {
        super(message, cause, true);
    }
}
