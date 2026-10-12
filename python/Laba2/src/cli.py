"""Команда подготовки корпусов практической работы № 2."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from .data import PreparationError as LegacyPreparationError
from .model import (
    ModelError,
    analyze,
    evaluate_corpus,
    load_model,
    model_summary,
    save_model,
    topics,
    train_model,
)
from .output import render_result
from .preparation import PreparationError, prepare_table, resolve_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Прототип семантического анализатора LSA")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="подготовить канонический CSV из таблицы")
    prepare.add_argument("--input", type=Path, help="путь исходного CSV, TSV или архива")
    prepare.add_argument("--output", type=Path, required=True, help="путь канонического CSV")
    prepare.add_argument("--config", type=Path, help="явный TOML подготовки")
    prepare.add_argument("--profile", choices=("rureviews", "lenta"), help="встроенный профиль")
    prepare.add_argument("--retained-source", type=Path, help="полный CSV Lenta.ru")
    prepare.add_argument("--format", choices=("text", "json", "pretty-json"), default="text")
    train = commands.add_parser("train", help="обучить и сохранить семантическую модель")
    train.add_argument("--corpus", type=Path, required=True, help="канонический CSV корпуса")
    train.add_argument("--model", type=Path, required=True, help="путь пакета модели joblib")
    train.add_argument("--format", choices=("text", "json", "pretty-json"), default="text")
    analyze_command = commands.add_parser("analyze", help="найти близкие документы")
    analyze_command.add_argument(
        "--model", type=Path, required=True, help="путь пакета модели joblib"
    )
    source = analyze_command.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="текст запроса")
    source.add_argument("--file", type=Path, help="UTF-8-файл с текстом запроса")
    analyze_command.add_argument("--top-k", type=int, default=5, help="число документов в выдаче")
    analyze_command.add_argument(
        "--format", choices=("text", "json", "pretty-json"), default="text"
    )
    topic_command = commands.add_parser("topics", help="показать нагрузки LSA-компонент")
    topic_command.add_argument(
        "--model", type=Path, required=True, help="путь пакета модели joblib"
    )
    topic_command.add_argument(
        "--top-terms", type=int, default=5, help="число терминов каждого знака"
    )
    topic_command.add_argument("--format", choices=("text", "json", "pretty-json"), default="text")
    evaluate = commands.add_parser("evaluate", help="сравнить TF-IDF и LSA на test-части")
    evaluate.add_argument("--corpus", type=Path, required=True, help="размеченный канонический CSV")
    evaluate.add_argument("--output", type=Path, required=True, help="путь JSON-протокола")
    evaluate.add_argument("--format", choices=("text", "json", "pretty-json"), default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            if args.profile is not None and args.config is not None:
                raise PreparationError("--profile и --config взаимоисключающие")
            config = _prepare_config(args.profile, args.input, args.config)
            input_path = args.input or _profile_input(args.profile)
            if input_path is None:
                raise PreparationError("Для prepare укажите --input")
            parameters = (
                {"retained_source": str(args.retained_source)}
                if args.retained_source is not None
                else None
            )
            result = prepare_table(input_path, args.output, config, adapter_parameters=parameters)
            print(render_result(result.as_dict(), args.format))
            return 0
        if args.command == "train":
            model = train_model(args.corpus)
            save_model(model, args.model)
            print(render_result(model_summary(model, args.model), args.format))
            return 0
        if args.command == "analyze":
            if args.file is not None:
                try:
                    text = args.file.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as error:
                    raise ModelError(
                        f"Не удалось прочитать UTF-8-файл {args.file}: {error}"
                    ) from error
            else:
                text = args.text
            print(render_result(analyze(load_model(args.model), text, args.top_k), args.format))
            return 0
        if args.command == "topics":
            print(render_result(topics(load_model(args.model), args.top_terms), args.format))
            return 0
        if args.command == "evaluate":
            protocol = evaluate_corpus(args.corpus)
            _write_json(args.output, protocol)
            # Вывод и сохранённый протокол описывают один эксперимент без разных сводок.
            print(render_result(protocol, args.format))
            return 0
    except (PreparationError, LegacyPreparationError, ModelError) as error:
        print(f"Ошибка данных: {error}", file=sys.stderr)
        return 1
    raise AssertionError(f"Необработанная команда: {args.command}")


def _prepare_config(profile: str | None, input_path: Path | None, explicit: Path | None) -> Path:
    if profile is None:
        if input_path is None:
            raise PreparationError("Для пользовательского источника укажите --input")
        return resolve_config(input_path, explicit)
    return Path(__file__).resolve().parents[1] / "profiles" / f"{profile}.toml"


def _profile_input(profile: str | None) -> Path | None:
    if profile == "rureviews":
        return Path("Laba1/data/rureviews.tsv")
    return None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    """Сохранить протокол атомарно, чтобы не оставить частичный JSON."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        temporary.replace(path)
    except (OSError, TypeError, ValueError) as error:
        cleanup_error: OSError | None = None
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as failed_cleanup:
                cleanup_error = failed_cleanup
        message = f"Не удалось сохранить JSON-протокол {path}: {error}"
        if cleanup_error is not None:
            message += f"; временный файл {temporary} не удалён: {cleanup_error}"
        raise ModelError(message) from error
