package ru.nlp.laba3.exception;

/** Базовая ошибка анализатора с явным выбором необходимости трассировки. */
public abstract class NlpException extends RuntimeException {

    /** Создаёт ошибку, сохраняя причину и выбранную политику стека. */
    protected NlpException(String message, Throwable cause, boolean writableStackTrace) {
        super(message, cause, true, writableStackTrace);
    }
}
