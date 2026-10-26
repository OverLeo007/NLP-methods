package ru.nlp.laba3.exception;

/** Ожидаемая ошибка пользовательских данных; место вызова не нужно для диагностики. */
public final class InvalidInputException extends NlpException {

    /** Создаёт отдельный экземпляр без сбора стека; сообщение содержит нарушенное условие. */
    public InvalidInputException(String message) {
        super(message, null, false);
    }
}
