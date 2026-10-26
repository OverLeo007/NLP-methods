package ru.nlp.laba3.preprocessing;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import org.springframework.stereotype.Component;
import org.tartarus.snowball.ext.RussianStemmer;

/** Общая нормализация документов и запросов: Snowball и стоп-слова Laba2. */
@Component
public class TextPreprocessor {

    private static final String PROFILE_SUFFIX = "snowball-ru-stop75-v1";

    private static final Pattern CYRILLIC = Pattern.compile("[а-я]+");

    private static final Pattern HYPHEN_BOUNDARY = Pattern.compile("(?=[-‑])|(?<=[-‑])");

    private final TextNormalizationChain normalizationChain;

    private final Set<String> stopwords;

    private final String resourceHash;

    /** Загружает встроенный стоп-список и нормализует его теми же правилами. */
    public TextPreprocessor(TextNormalizationChain normalizationChain) throws IOException {
        this.normalizationChain = normalizationChain;

        try (var input = TextPreprocessor.class.getResourceAsStream("/stopwords_ru.txt")) {

            if (input == null) {
                throw new IOException("Отсутствует ресурс stopwords_ru.txt");
            }

            byte[] bytes = input.readAllBytes();
            resourceHash = sha256(bytes);
            stopwords = new String(bytes, StandardCharsets.UTF_8)
                    .lines()
                    .map(String::strip)
                    .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                    .map(line -> stem(TextTokenizer.tokens(normalizationChain.normalize(line)).getFirst()))
                    .collect(Collectors.toUnmodifiableSet());
        }
    }

    /** Возвращает нормализованные токены без замены синонимов. */
    public List<String> tokens(String text) {
        return TextTokenizer.tokens(normalizationChain.normalize(text)).stream()
                .map(TextPreprocessor::stem)
                .filter(token -> !stopwords.contains(token))
                .toList();
    }

    /** Контрольная сумма ресурса для проверки совместимости сохранённой модели. */
    public String resourceHash() {
        return resourceHash;
    }

    /** Проверяет наличие слов или чисел до удаления стоп-слов. */
    public boolean hasTokens(String text) {
        return !TextTokenizer.tokens(normalizationChain.normalize(text)).isEmpty();
    }

    /** Возвращает профиль с учётом состава и порядка правил нормализации. */
    public String profile() {
        return normalizationChain.profile() + "-" + PROFILE_SUFFIX;
    }

    private static String stem(String token) {
        // Как в Laba2: стемминг только целиком кириллических сегментов вокруг дефиса.
        var parts = HYPHEN_BOUNDARY.split(token);
        var result = new StringBuilder();

        for (String part : parts) {
            if (CYRILLIC.matcher(part).matches()) {
                var stemmer = new RussianStemmer();
                stemmer.setCurrent(part);
                stemmer.stem();
                result.append(stemmer.getCurrent());
            } else {
                result.append(part);
            }
        }

        return result.toString();
    }

    private static String sha256(byte[] value) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value));
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 недоступен", exception);
        }
    }
}
