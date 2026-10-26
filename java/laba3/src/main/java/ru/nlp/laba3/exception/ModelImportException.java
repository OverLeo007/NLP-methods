package ru.nlp.laba3.exception;

/** Ошибка чтения, целостности или восстановления модели. Причина и трассировка сохраняются. */
public final class ModelImportException extends NlpException {

    /** Создаёт ошибку проверки с диагностической трассировкой. */
    public ModelImportException(String message) {
        this(message, null);
    }

    /** Сохраняет исходную ошибку для диагностики сбоя операции. */
    public ModelImportException(String message, Throwable cause) {
        super(message, cause, true);
    }
}
