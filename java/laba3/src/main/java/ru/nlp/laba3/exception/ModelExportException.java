package ru.nlp.laba3.exception;

/** Ошибка записи файлов модели. Причина и трассировка сохраняются. */
public final class ModelExportException extends NlpException {

    /** Создаёт ошибку проверки с диагностической трассировкой. */
    public ModelExportException(String message) {
        this(message, null);
    }

    /** Сохраняет исходную ошибку для диагностики сбоя операции. */
    public ModelExportException(String message, Throwable cause) {
        super(message, cause, true);
    }
}
