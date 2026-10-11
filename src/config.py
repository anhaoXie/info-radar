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
        return json.loads(read_text_smart(path))
    except json.JSONDecodeError as exc:
        raise ValueError(f"配置文件不是合法 JSON：{path}\n  {exc}") from exc


def read_text_smart(path: Path) -> str:
    """读文本文件，对编码宽容一点。

    【为什么不能只用 utf-8】

    中文 Windows 上，记事本和不少编辑器默认把文件存成 GBK。
    而本项目里恰有两个文件是【明确希望用户手工编辑】的：

        eval/labels.txt        标注表
        eval/eval_set.jsonl    评估集

    用户用记事本改一下、保存，再跑命令 —— 如果这里用 utf-8 硬读，
    得到的是一个 UnicodeDecodeError 的英文报错栈。
    他完全看不出"这是编码问题"，更看不出该怎么做。

    （这个 bug 是被自己的测试意外挖出来的：我把测试数据用
      PowerShell 的 Add-Content 追加进去，它按 GBK 写，
      于是下一次 compile 直接抛栈。而中文 Windows 用户的编辑器
      默认干的正是同一件事。）

    依次尝试 utf-8-sig（能吃掉 BOM）、utf-8、gbk；都不行就带替换地读，
    至少让程序继续跑，并让内容里的乱码自己暴露出来 ——
    比一个看不出所以然的报错栈好。
    """
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


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
