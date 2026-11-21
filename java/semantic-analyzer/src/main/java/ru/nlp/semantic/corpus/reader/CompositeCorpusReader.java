package ru.nlp.semantic.corpus.reader;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Predicate;

import org.springframework.context.annotation.Primary;
import org.springframework.stereotype.Component;

import ru.nlp.semantic.corpus.dto.Document;
import ru.nlp.semantic.exception.InvalidInputException;

import lombok.RequiredArgsConstructor;

/** Выбирает реализацию чтения по формату пути; не вмешивается в состав документов. */
@Component
@Primary
@RequiredArgsConstructor
public class CompositeCorpusReader implements CorpusReader {

    private final List<CorpusReader> readers;

    /** Проверяет, поддерживается ли путь хотя бы одним читателем. */
    @Override
    public boolean supports(Path input) {
        return readers.stream().anyMatch(reader -> reader.supports(input));
    }

    /** Делегирует чтение через общий контракт соответствующего формата. */
    @Override
    public long scan(Path input, Predicate<Document> consumer) throws IOException {
        var reader = readers.stream().filter(candidate -> candidate.supports(input)).findFirst()
                .orElseThrow(() -> new InvalidInputException("Ожидается CSV, JSONL, ZIP РИА или папка TXT: " + input));

        return reader.scan(input, consumer);
    }
}
