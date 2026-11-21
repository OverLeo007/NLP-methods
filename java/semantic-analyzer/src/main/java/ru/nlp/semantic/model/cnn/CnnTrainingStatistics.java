package ru.nlp.semantic.model.cnn;

import java.util.List;

/** Диагностика обучения и отдельная validation-история, если разделение было запрошено. */
public record CnnTrainingStatistics(
        /** Число различных пригодных обучающих документов. */
        int trainingDocuments,
        /** Число выполненных обновлений весов. */
        int updates,
        /** Средняя потеря по пакетам каждой эпохи до соответствующих обновлений. */
        List<Double> epochLosses,
        /** Потеря на фиксированном обучающем пакете до обучения; это не validation. */
        double initialDiagnosticLoss,
        /** Потеря на том же пакете после обучения. */
        double finalDiagnosticLoss,
        /** L2-норма изменения всех обучаемых весов. */
        double parameterChangeL2,
        /** Число документов в фактически оценённых validation-пакетах; 0 без разделения. */
        int validationDocuments,
        /** Потеря после каждой эпохи на одних и тех же отложенных парах. */
        List<Double> validationLosses,
        /** Первая эпоха с минимальной validation-потерей; 0 без validation. Веса остаются последней эпохи. */
        int bestValidationEpoch
) {

    /** Защищает историю и отклоняет неполную или неконечную диагностику. */
    public CnnTrainingStatistics {
        epochLosses = List.copyOf(epochLosses);
        // В комплектах прототипа этих необязательных полей ещё не было.
        validationLosses = validationLosses == null ? List.of() : List.copyOf(validationLosses);

        if (
            trainingDocuments < 2 || updates < 1 || epochLosses.isEmpty()
                    || epochLosses.stream().anyMatch(value -> !Double.isFinite(value) || value < 0)
                    || !Double.isFinite(initialDiagnosticLoss) || initialDiagnosticLoss < 0
                    || !Double.isFinite(finalDiagnosticLoss) || finalDiagnosticLoss < 0
                    || !Double.isFinite(parameterChangeL2) || parameterChangeL2 <= 0
                    || validationDocuments < 0
                    || (validationDocuments == 0 && (!validationLosses.isEmpty() || bestValidationEpoch != 0))
                    || (validationDocuments > 0 && (validationDocuments < 2
                            || validationLosses.size() != epochLosses.size()
                            || bestValidationEpoch < 1 || bestValidationEpoch > validationLosses.size()))
                    || validationLosses.stream().anyMatch(value -> !Double.isFinite(value) || value < 0)
        ) {
            throw new IllegalArgumentException("CNN: некорректная статистика обучения");
        }
    }
}
