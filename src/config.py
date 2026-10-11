"""配置读取。

全项目只有这一处知道配置文件在哪。这样"配置文件搬家了"只需要改一个地方。
顺带一提：这就是"配置与代码分离"最朴素的样子 ——
改行为不用改代码，于是改行为这件事变得便宜，于是你才愿意做实验。
"""

from __future__ import annotations

import json
from pathlib import Path

from .secrets import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "config"


def config_path(name: str) -> Path:
    return CONFIG_DIR / (name if name.endswith(".json") else f"{name}.json")


def load_json(name: str) -> dict:
    path = config_path(name)
    if not path.exists():
        raise FileNotFoundError(f"找不到配置文件：{path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件不是合法 JSON：{path}\n  {exc}") from exc


def save_json(name: str, data: dict) -> Path:
    path = config_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_policy() -> dict:
    return load_json("policy")


def load_providers() -> dict:
    return load_json("providers")


def load_task() -> dict:
    return load_json("task")


def load_sources() -> dict:
    return load_json("sources")
