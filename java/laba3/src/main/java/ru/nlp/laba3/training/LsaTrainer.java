package ru.nlp.laba3.training;

import static org.bytedeco.openblas.global.openblas.LAPACKE_dgesvd;
import static org.bytedeco.openblas.global.openblas.LAPACK_ROW_MAJOR;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import org.bytedeco.javacpp.DoublePointer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Component;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.exception.ModelTrainingException;
import ru.nlp.laba3.model.analysis.Algorithm;
import ru.nlp.laba3.model.lsa.LsaModel;
import ru.nlp.laba3.model.lsa.LsaParameters;
import ru.nlp.laba3.runtime.StageTimings;
import ru.nlp.laba3.vectorization.TfIdf;

import lombok.extern.slf4j.Slf4j;

/** Точный библиотечный SVD с усечением и ограничением размера матрицы. */
@Component
@Slf4j
public class LsaTrainer extends AbstractModelTrainer<LsaParameters, LsaModel> {

    private static final long MAX_MATRIX_CELLS = 20_000_000;

    /** Задаёт алгоритм для общего сценария нативного обучения. */
    public LsaTrainer() {
        super(Algorithm.lsa);
    }

    /** Возвращает число потоков обучения LSA. */
    @Override
    protected int workers(LsaParameters parameters) {
        return parameters.workers();
    }

    /** Возвращает фактический размер словаря LSA. */
    @Override
    protected int vocabularySize(LsaModel model) {
        return model.vocabulary().size();
    }

    /** Возвращает фактическую размерность LSA. */
    @Override
    protected int dimensions(LsaModel model) {
        return model.singularValues().length;
    }

    /** Обучает специфичное состояние LSA внутри общего нативного контекста. */
    @Override
    protected LsaModel fit(List<List<String>> documents, LsaParameters parameters) {
        long started = System.nanoTime();
        var vocabulary = selectVocabulary(documents, parameters.maxFeatures());
        validateSize(documents.size(), vocabulary.size());
        var fittedIdf = TfIdf.fitIdf(documents);
        double[] idf = vocabulary.stream().mapToDouble(fittedIdf::get).toArray();
        var positions = termPositions(vocabulary);
        int rank = Math.min(documents.size(), vocabulary.size());
        int dimensions = Math.min(parameters.dimensions(), rank - 1);
        log.info(
                "[{}] TF-IDF {} × {}, SVD-компонент={}",
                algorithm.displayName(),
                documents.size(),
                vocabulary.size(),
                dimensions
        );

        try (
                var matrix = Nd4j.zeros(DataType.DOUBLE, vocabulary.size(), documents.size());
                var singular = Nd4j.create(DataType.DOUBLE, rank);
                var left = Nd4j.create(DataType.DOUBLE, vocabulary.size(), rank);
                var superb = Nd4j.create(DataType.DOUBLE, rank)
        ) {
            fillMatrix(matrix, documents, positions, idf);
            fitModel(matrix, singular, left, superb, started);
            var model = createModel(vocabulary, idf, parameters, singular, left, dimensions);

            return model;
        }
    }

    private List<String> selectVocabulary(List<List<String>> documents, int maxFeatures) {
        var counts = new HashMap<String, Long>();
        documents.forEach(tokens -> tokens.forEach(token -> counts.merge(token, 1L, Long::sum)));

        return counts.entrySet().stream()
                .sorted(
                        Map.Entry.<String, Long>comparingByValue(Comparator.reverseOrder())
                                .thenComparing(Map.Entry.comparingByKey())
                )
                .limit(maxFeatures).map(Map.Entry::getKey).sorted().toList();
    }

    private void validateSize(int documents, int terms) {
        if (documents < 2 || terms < 2) {
            throw new InvalidInputException("LSA нужны минимум 2 документа и 2 различных нормализованных термина");
        }

        if ((long) documents * terms > MAX_MATRIX_CELLS) {
            throw new InvalidInputException(
                    "Плотный SVD LSA ограничен 20000000 ячеек: уменьшите --limit или --max-features"
            );
        }
    }

    private Map<String, Integer> termPositions(List<String> vocabulary) {
        var positions = new HashMap<String, Integer>();

        for (int i = 0; i < vocabulary.size(); i++) {
            positions.put(vocabulary.get(i), i);
        }

        return positions;
    }

    private void fillMatrix(
            INDArray matrix,
            List<List<String>> documents,
            Map<String, Integer> positions,
            double[] idf
    ) {
        for (int i = 0; i < documents.size(); i++) {
            int column = i;
            TfIdf.sublinearVector(documents.get(i), positions, idf)
                    .forEach((row, value) -> matrix.putScalar(row, column, value));
        }
    }

    private void fitModel(INDArray matrix, INDArray singular, INDArray left, INDArray superb, long started) {
        try (var heartbeat = Executors.newSingleThreadScheduledExecutor(runnable -> {
            var thread = new Thread(runnable, algorithm.name() + "-progress");
            thread.setDaemon(true);

            return thread;
        })) {
            heartbeat.scheduleAtFixedRate(
                    () -> log.info(
                            "[{}] SVD активно: документов={}, терминов={}, прошло={} с",
                            algorithm.displayName(),
                            matrix.columns(),
                            matrix.rows(),
                            StageTimings.elapsed(started)
                    ),
                    5,
                    5,
                    TimeUnit.SECONDS
            );

            try {
                decompose(matrix, singular, left, superb);
            } finally {
                heartbeat.shutdownNow();
            }
        }
    }

    private void decompose(INDArray matrix, INDArray singular, INDArray left, INDArray superb) {
        // Thin U транспонированной матрицы содержит термовые компоненты.
        // LAPACK использует те же CPU-библиотеки, что ND4J; VT для поиска не нужен.
        int status = LAPACKE_dgesvd(
                LAPACK_ROW_MAJOR,
                (byte) 'S',
                (byte) 'N',
                matrix.rows(),
                matrix.columns(),
                (DoublePointer) matrix.data().addressPointer(),
                matrix.columns(),
                (DoublePointer) singular.data().addressPointer(),
                (DoublePointer) left.data().addressPointer(),
                left.columns(),
                (DoublePointer) null,
                1,
                (DoublePointer) superb.data().addressPointer()
        );

        if (status != 0) {
            throw new ModelTrainingException("LSA: LAPACK dgesvd завершился с кодом " + status);
        }
    }

    private LsaModel createModel(
            List<String> vocabulary,
            double[] idf,
            LsaParameters parameters,
            INDArray singular,
            INDArray left,
            int dimensions
    ) {
        double[][] projection = new double[vocabulary.size()][dimensions];
        double[] values = new double[dimensions];

        for (int j = 0; j < dimensions; j++) {
            values[j] = singular.getDouble(j);

            if (!Double.isFinite(values[j])) {
                throw new ModelTrainingException("LSA: SVD вернул нечисловые сингулярные числа");
            }

            for (int i = 0; i < vocabulary.size(); i++) {
                projection[i][j] = left.getDouble(i, j);

                if (!Double.isFinite(projection[i][j])) {
                    throw new ModelTrainingException("LSA: SVD вернул нечисловую проекцию");
                }
            }
        }

        return new LsaModel(vocabulary, idf, projection, values, parameters);
    }
}
