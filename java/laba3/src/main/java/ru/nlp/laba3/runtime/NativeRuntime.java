package ru.nlp.laba3.runtime;

import java.io.IOException;
import java.util.function.Supplier;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import ru.nlp.laba3.exception.InvalidInputException;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/** Общая для обучения и поиска настройка CPU-потоков. */
@Slf4j
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class NativeRuntime {

    /** Выполняет нативный этап с настройкой потоков, исключая другие нативные этапы процесса. */
    public static synchronized <T> T withThreads(int workers, NativeOperation<T> operation) throws IOException {
        if (workers < 1) {
            throw new InvalidInputException("Число потоков должно быть >= 1");
        }
        configureThreads(workers);

        return operation.execute();
    }

    /** Защищает чтение нативных векторов от одновременной смены общих настроек ND4J. */
    public static synchronized <T> T withLock(Supplier<T> operation) {
        return operation.get();
    }

    /** Нативный этап, который может завершиться ошибкой чтения или обучения. */
    @FunctionalInterface
    public interface NativeOperation<T> {

        /** Выполняет этап внутри общего монитора нативной среды. */
        T execute() throws IOException;
    }

    private static void configureThreads(int workers) {
        Nd4j.getEnvironment().setMaxThreads(workers);
        NativeOpsHolder.getInstance().getDeviceNativeOps().setOmpNumThreads(workers);
        log.info(
                "Потоки: ND4J={}, OpenMP={}",
                workers,
                NativeOpsHolder.getInstance().getDeviceNativeOps().ompGetMaxThreads()
        );
    }
}
