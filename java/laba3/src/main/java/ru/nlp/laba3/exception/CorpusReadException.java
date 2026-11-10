package ru.nlp.laba3.exception;

/** Ошибка чтения или разбора источника документов. Причина и трассировка сохраняются. */
public final class CorpusReadException extends NlpException {

    /** Создаёт ошибку проверки с диагностической трассировкой. */
    public CorpusReadException(String message) {
        this(message, null);
    }

    /** Сохраняет исходную ошибку для диагностики сбоя операции. */
    public CorpusReadException(String message, Throwable cause) {
        super(message, cause, true);
    }
}
