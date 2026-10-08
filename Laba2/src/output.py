"""Единое представление успешных результатов CLI."""

from __future__ import annotations

import json
from typing import Any


def render_result(payload: dict[str, Any], format_name: str) -> str:
    """Сериализовать тот же результат в text, compact JSON или pretty JSON."""
    if format_name == "json":
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if format_name == "pretty-json":
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    lines = [f"{key}: {value}" for key, value in payload.items()]
    return "\n".join(lines)
