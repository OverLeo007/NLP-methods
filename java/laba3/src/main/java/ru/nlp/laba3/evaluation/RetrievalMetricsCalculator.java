package ru.nlp.laba3.evaluation;

import java.util.HashMap;
import java.util.List;
import java.util.Set;

import ru.nlp.laba3.exception.InvalidInputException;
import ru.nlp.laba3.model.evaluation.EvaluationQuery;
import ru.nlp.laba3.model.evaluation.QueryEvaluation;
import ru.nlp.laba3.model.evaluation.RetrievalMetrics;
import ru.nlp.laba3.model.search.SearchHit;
import ru.nlp.laba3.model.search.SearchResult;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/** Формулы оценки общие для всех способов обучения и всех статусов запросов. */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class RetrievalMetricsCalculator {

    /** Считает Recall@5 и reciprocal rank@10 по положительным ID одного запроса. */
    public static QueryEvaluation score(
            EvaluationQuery query,
            Set<String> relevantIds,
            Set<String> zeroDocumentIds,
            SearchResult result,
            double seconds
    ) {
        if (relevantIds.isEmpty()) {
            throw new InvalidInputException("Для оценки нужна положительная разметка: " + query.id());
        }
        var hits = result.hits().stream().limit(10).toList();
        long found = hits.stream().limit(5).map(SearchHit::id).distinct().filter(relevantIds::contains).count();
        double recall = (double) found / relevantIds.size();
        double reciprocalRank = 0;

        for (int i = 0; i < hits.size(); i++) {
            if (relevantIds.contains(hits.get(i).id())) {
                reciprocalRank = 1.0 / (i + 1);
                break;
            }
        }

        var zeros = relevantIds.stream().filter(zeroDocumentIds::contains).sorted().toList();
        String reason = "none";

        if (recall < 1) {
            reason = !"ok".equals(result.status())
                    ? result.status()
                    : zeros.isEmpty() ? "ranking" : "zero_relevant_document";
        }

        return new QueryEvaluation(
                query.id(),
                query.text(),
                result.status(),
                result.knownTokens(),
                result.totalTokens(),
                relevantIds.stream().sorted().toList(),
                zeros,
                hits,
                recall,
                reciprocalRank,
                reason,
                seconds
        );
    }

    /** Усредняет вклады по всем запросам, включая OOV, пустые и нулевые. */
    public static RetrievalMetrics aggregate(List<QueryEvaluation> results) {
        if (results.isEmpty()) {
            throw new InvalidInputException("Оценочный набор запросов пуст");
        }
        var statuses = new HashMap<String, Long>();
        var reasons = new HashMap<String, Long>();
        long known = 0;
        long total = 0;

        for (var result : results) {
            statuses.merge(result.status(), 1L, Long::sum);

            if (!"none".equals(result.recallLossReason())) {
                reasons.merge(result.recallLossReason(), 1L, Long::sum);
            }
            known += result.knownTokens();
            total += result.totalTokens();
        }

        double seconds = results.stream().mapToDouble(QueryEvaluation::searchSeconds).sum();
        var sortedSeconds = results.stream().mapToDouble(QueryEvaluation::searchSeconds).sorted().toArray();

        return new RetrievalMetrics(
                results.size(),
                results.stream().mapToDouble(QueryEvaluation::recallAt5).average().orElseThrow(),
                results.stream().mapToDouble(QueryEvaluation::reciprocalRankAt10).average().orElseThrow(),
                results.stream().filter(row -> row.recallAt5() > 0).count(),
                results.stream().filter(row -> row.reciprocalRankAt10() > 0).count(),
                statuses,
                results.stream().filter(row -> row.knownTokens() > 0 && row.knownTokens() < row.totalTokens()).count(),
                known,
                total,
                total == 0 ? 0 : (double) known / total,
                results.stream().filter(row -> !row.zeroRelevantIds().isEmpty()).count(),
                reasons,
                seconds,
                seconds / results.size(),
                sortedSeconds[(int) Math.ceil(results.size() * 0.95) - 1]
        );
    }
}
