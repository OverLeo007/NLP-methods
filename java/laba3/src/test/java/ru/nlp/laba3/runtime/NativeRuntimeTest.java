package ru.nlp.laba3.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

import org.junit.jupiter.api.Test;

class NativeRuntimeTest {

    @Test
    void nativeOperationsExcludeEachOtherAndReleaseMonitorAfterFailure() throws Exception {
        var firstEntered = new CountDownLatch(1);
        var releaseFirst = new CountDownLatch(1);
        var secondStarted = new CountDownLatch(1);
        var secondEntered = new AtomicBoolean();

        try (var executor = Executors.newFixedThreadPool(2)) {
            var first = executor.submit(() -> NativeRuntime.withLock(() -> {
                firstEntered.countDown();

                try {
                    if (!releaseFirst.await(5, TimeUnit.SECONDS)) {
                        throw new AssertionError("Не получен сигнал освобождения");
                    }
                } catch (InterruptedException exception) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(exception);
                }

                return true;
            }));

            try {
                assertThat(firstEntered.await(5, TimeUnit.SECONDS)).isTrue();
                var second = executor.submit(() -> {
                    secondStarted.countDown();

                    return NativeRuntime.withLock(() -> {
                        secondEntered.set(true);

                        return true;
                    });
                });
                assertThat(secondStarted.await(5, TimeUnit.SECONDS)).isTrue();
                assertThatThrownByTimeout(second);
                assertThat(secondEntered).isFalse();
                releaseFirst.countDown();
                assertThat(first.get(5, TimeUnit.SECONDS)).isTrue();
                assertThat(second.get(5, TimeUnit.SECONDS)).isTrue();
                var failure = executor.submit(() -> NativeRuntime.withLock(() -> {
                    throw new IllegalStateException("probe");
                }));
                assertThatThrownBy(() -> failure.get(5, TimeUnit.SECONDS))
                        .hasCauseInstanceOf(IllegalStateException.class);
                assertThat(executor.submit(() -> NativeRuntime.withLock(() -> true)).get(5, TimeUnit.SECONDS)).isTrue();
            } finally {
                releaseFirst.countDown();
            }
        }
    }

    private static void assertThatThrownByTimeout(Future<?> future) {
        assertThatThrownBy(() -> future.get(100, TimeUnit.MILLISECONDS))
                .isInstanceOf(TimeoutException.class);
    }
}
