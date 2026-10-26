package ru.nlp.laba3;

import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;

import org.slf4j.Logger;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.logging.LogLevel;
import org.springframework.boot.logging.LoggingSystem;

import picocli.CommandLine;

/** Точка входа семантического анализатора лабораторной работы №3. */
@SpringBootApplication
public class Laba3Application {

    /** Запускает CLI с фабрикой команд и внедрением зависимостей Spring. */
    static void main(String[] args) {
        System.exit(execute(args));
    }

    /** Выполняет один вызов CLI, закрывает контекст и возвращает код завершения. */
    static int execute(String[] args) {
        try (var context = SpringApplication.run(Laba3Application.class, args)) {
            var factory = context.getBean(CommandLine.IFactory.class);
            var rootCommand = context.getBean(Laba3Command.class);

            return new CommandLine(rootCommand, factory)
                    .setOut(new PrintWriter(System.out, true, StandardCharsets.UTF_8))
                    .setErr(new PrintWriter(System.err, true, StandardCharsets.UTF_8))
                    .setExecutionStrategy(parseResult -> {
                        configureLogging(rootCommand.isVerbose());

                        return new CommandLine.RunLast().execute(parseResult);
                    })
                    .setExecutionExceptionHandler(
                            (exception, command, _) -> {
                                command.getErr().println("Ошибка: " + exception.getMessage());

                                return 1;
                            }
                    )
                    .execute(args);
        }
    }

    private static void configureLogging(boolean verbose) {
        LoggingSystem.get(Laba3Application.class.getClassLoader())
                .setLogLevel(Logger.ROOT_LOGGER_NAME, verbose ? LogLevel.INFO : LogLevel.OFF);
    }
}
