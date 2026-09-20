"""Внутренняя реализация интерфейса командной строки лабораторной работы № 1."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from .data import (
    DataValidationError,
    prepare_rureviews,
    sha256_file,
)
from .experiment import (
    SELECTED_CONFIG,
    ExperimentError,
    TestEvaluation,
    configuration_id,
    evaluate_selected_model,
    train_selected_model,
)
from .model import ModelPackageError, build_metadata, load_model_package, save_model_package
from .normalization import NormalizationConfig, NormalizationError, TextNormalizer
from .tokenizer import tokenize


class CliInputError(ValueError):
    """Ошибка обработки пользовательского ввода для анализа."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Анализ тональности русскоязычных отзывов")
    commands = parser.add_subparsers(dest="command", required=True)
    vocabulary = commands.add_parser("vocabulary", help="показать токены и нормализацию")
    vocabulary.add_argument("--text", required=True, help="текст для демонстрации")
    vocabulary.add_argument(
        "--method",
        choices=("none", "stem", "lemma"),
        default="none",
        help="морфологическая ветвь",
    )
    vocabulary.add_argument("--stopwords", action="store_true", help="удалить стоп-слова")
    vocabulary.add_argument("--synonyms", action="store_true", help="свести синонимы")
    train = commands.add_parser("train", help="обучить и сохранить выбранную модель")
    train.add_argument(
        "--artifacts", type=Path, required=True, help="каталог для локального пакета модели"
    )
    train.add_argument("--verbose", action="store_true", help="показать этапы обучения в stderr")
    analyze = commands.add_parser("analyze", help="определить тональность пользовательского текста")
    analyze.add_argument("--model", type=Path, required=True, help="путь к пакету model.joblib")
    input_source = analyze.add_mutually_exclusive_group(required=True)
    input_source.add_argument("--text", help="строка для анализа")
    input_source.add_argument("--file", type=Path, help="UTF-8-файл с текстом для анализа")
    return parser


def _configuration_payload(config: NormalizationConfig) -> dict[str, object]:
    return {
        "method": config.method,
        "stopwords": config.remove_stopwords,
        "synonyms": config.use_synonyms,
    }


def _read_analysis_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    try:
        return args.file.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise CliInputError(f"Не удалось прочитать файл {args.file} как UTF-8: {error}") from error
    except OSError as error:
        raise CliInputError(f"Не удалось прочитать файл {args.file}: {error}") from error


def _analyze(args: argparse.Namespace) -> None:
    text = _read_analysis_text(args)
    if not tokenize(text):
        raise CliInputError("Текст для анализа не содержит токенов")
    package = load_model_package(args.model)
    probabilities = package.model.predict_probabilities([text])[0]
    winning_index = max(
        range(len(package.model.class_order)), key=lambda index: probabilities[index]
    )
    print(
        json.dumps(
            {
                "configuration": _configuration_payload(package.model.config),
                "label": package.model.class_order[winning_index],
                "probabilities": dict(zip(package.model.class_order, probabilities, strict=True)),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _train(args: argparse.Namespace) -> None:
    corpus_path = Path(__file__).resolve().parents[1] / "data" / "rureviews.tsv"
    _progress(args.verbose, "[1/5] Чтение и очистка поставляемого корпуса RuReviews")
    prepared = prepare_rureviews(corpus_path)
    _progress(
        args.verbose,
        "[2/5] Части корпуса: "
        f"train={len(prepared.split.train)}, validation={len(prepared.split.validation)}, "
        f"test={len(prepared.split.test)}",
    )
    _progress(args.verbose, f"[3/5] Обучение {configuration_id(SELECTED_CONFIG)}")
    training = train_selected_model(prepared.split.train)
    _progress(args.verbose, "[4/5] Оценка на отложенной test-части")
    evaluation = evaluate_selected_model(training.model, prepared.split.test)
    _progress(
        args.verbose,
        f"Test: accuracy={evaluation.accuracy:.6f}, macro-F1={evaluation.macro_f1:.6f}",
    )
    model_path = args.artifacts / "model.joblib"
    metrics_path = args.artifacts / "metrics.json"
    predictions_path = args.artifacts / "predictions.json"
    metadata = build_metadata(corpus_sha256=sha256_file(corpus_path))
    save_model_package(model_path, training.model, metadata)
    _write_json(
        metrics_path,
        {
            "configuration": _configuration_payload(training.model.config),
            "test": evaluation.as_dict(),
        },
    )
    _write_json(
        predictions_path,
        {
            "class_order": list(training.model.class_order),
            "predictions": list(evaluation.predictions),
        },
    )
    image_path = _save_confusion_matrix(evaluation)
    _progress(
        args.verbose,
        "[5/5] Артефакты: "
        f"model={model_path}, metrics={metrics_path}, predictions={predictions_path}, "
        f"confusion_matrix={image_path}",
    )
    print(
        json.dumps(
            {
                "configuration": _configuration_payload(training.model.config),
                "model": str(model_path),
                "test": evaluation.as_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _progress(enabled: bool, message: str) -> None:
    """Напечатать компактный этап длительной операции только по запросу."""
    if enabled:
        print(message, file=sys.stderr)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Атомарно записать результат эксперимента без частичного JSON при сбое."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        prefix=f".{path.name}.",
        dir=path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(payload, temporary, ensure_ascii=False, indent=2, sort_keys=True)
        temporary.write("\n")
    temporary_path.replace(path)


def _save_confusion_matrix(evaluation: TestEvaluation) -> Path:
    """Построить итоговую матрицу ошибок в публичном порядке классов."""
    import matplotlib.pyplot as plt

    image_path = Path(__file__).resolve().parents[1] / "images" / "confusion_matrix.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(figsize=(6, 5))
    rendered = axes.imshow(evaluation.confusion_matrix, cmap="Blues")
    axes.figure.colorbar(rendered, ax=axes)
    axes.set(
        xticks=range(len(evaluation.confusion_matrix)),
        yticks=range(len(evaluation.confusion_matrix)),
        xticklabels=experiment_labels(),
        yticklabels=experiment_labels(),
        xlabel="Предсказанная метка",
        ylabel="Фактическая метка",
        title="Матрица ошибок выбранной модели",
    )
    for row_index, row in enumerate(evaluation.confusion_matrix):
        for column_index, value in enumerate(row):
            axes.text(column_index, row_index, str(value), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    return image_path


def experiment_labels() -> tuple[str, ...]:
    """Вернуть фиксированный публичный порядок классов для матрицы ошибок."""
    return "negative", "neutral", "positive"


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "vocabulary":
            config = NormalizationConfig(
                method=args.method,
                remove_stopwords=args.stopwords,
                use_synonyms=args.synonyms,
            )
            result = TextNormalizer(config).normalize(args.text)
            print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
            return 0
        if args.command == "train":
            _train(args)
            return 0
        if args.command == "analyze":
            _analyze(args)
            return 0
    except DataValidationError as error:
        print(f"Ошибка данных: {error}", file=sys.stderr)
        return 1
    except ExperimentError as error:
        print(f"Ошибка эксперимента: {error}", file=sys.stderr)
        return 1
    except ModelPackageError as error:
        print(f"Ошибка модели: {error}", file=sys.stderr)
        return 1
    except NormalizationError as error:
        print(f"Ошибка нормализации: {error}", file=sys.stderr)
        return 1
    except CliInputError as error:
        print(f"Ошибка ввода: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"Необработанная команда: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
