package ru.nlp.laba3.corpus.reader;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Iterator;
import java.util.Locale;
import java.util.stream.Stream;

import org.springframework.stereotype.Component;

import ru.nlp.laba3.model.corpus.Document;

/** Читает папку UTF-8 TXT рекурсивно в порядке путей. */
@Component
public class TxtCorpusReader extends AbstractCorpusReader {

    /** Распознаёт папку документов, включая папку с расширением в имени. */
    @Override
    public boolean supports(Path input) {
        return Files.isDirectory(input);
    }

    /** Открывает обход каталога; тексты файлов читаются только по требованию. */
    @Override
    protected DocumentCursor open(Path input) throws IOException {
        Stream<Path> paths = Files.walk(input).filter(Files::isRegularFile)
                .filter(path -> path.toString().toLowerCase(Locale.ROOT).endsWith(".txt")).sorted();
        Iterator<Path> files = paths.iterator();

        return new DocumentCursor() {

            @Override
            public Document next() throws IOException {
                if (!files.hasNext()) {
                    return null;
                }
                Path file = files.next();

                return new Document(
                        input.relativize(file).toString().replace('\\', '/'),
                        file.getFileName().toString(),
                        Files.readString(file, StandardCharsets.UTF_8)
                );
            }

            @Override
            public void close() {
                paths.close();
            }
        };
    }
}
