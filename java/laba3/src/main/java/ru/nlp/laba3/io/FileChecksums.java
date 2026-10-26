package ru.nlp.laba3.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Контроль целостности файлов моделей и исходных данных. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class FileChecksums {

    /** Вычисляет SHA-256 потоково, без размещения всего файла в памяти. */
    public static String hash(Path path) throws IOException {
        try {
            var digest = MessageDigest.getInstance("SHA-256");

            try (var input = Files.newInputStream(path)) {
                byte[] buffer = new byte[65536];
                int length;

                while ((length = input.read(buffer)) != -1) {
                    digest.update(buffer, 0, length);
                }
            }

            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 недоступен", exception);
        }
    }
}
