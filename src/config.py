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


class ConfigError(Exception):
    """配置文件有问题。

    单独定义一种异常，是为了让它在最外层被翻译成一句人话，
    而不是变成一段 Python 报错栈。

    为什么值得这么做：S3 阶段你会反复手改 config/task.json 的 few_shot。
    一次逗号手滑，原来会得到 20 行英文栈 —— 你会以为"系统坏了"，
    而实际上只是少了个引号，而且【改错配置文件不会有任何副作用】，
    因为读不进来就什么都不会执行。
    """


def load_json(name: str) -> dict:
    path = config_path(name)
    if not path.exists():
        raise ConfigError(f"找不到配置文件：{path}")
    text = read_text_smart(path)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # 光说"第 46 行"还得自己去找。把那一行和指针一起打出来。
        lines = text.splitlines()
        bad_line = lines[exc.lineno - 1] if 0 < exc.lineno <= len(lines) else ""
        pointer = " " * max(0, exc.colno - 1) + "^"
        raise ConfigError(
            f"{path}\n"
            f"  第 {exc.lineno} 行第 {exc.colno} 列：{exc.msg}\n"
            f"\n"
            f"  {bad_line}\n"
            f"  {pointer}\n"
        ) from exc


def main_guard(fn):
    """把一个"配置读不进来"的异常，翻译成用户看得懂的三句话。

    不做这件事的后果很具体：用户照着自己的学习路线去改配置，
    手滑一次，得到一段英文报错栈 —— 他会认为"我把系统弄坏了"，
    然后不敢再改配置。而整套系统的设计前提，恰恰是
    **改配置是安全的、便宜的、应该经常做的**。
    """

    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ConfigError as exc:
            print()
            print("  " + "=" * 64)
            print("  配置文件有问题")
            print("  " + "=" * 64)
            print()
            for line in str(exc).splitlines():
                print(f"  {line}" if line.strip() else "")
            print("  配置文件是 JSON。最常见的手滑：多一个逗号、少一个引号、少一个括号。")
            print("  改完再跑一次就行 —— 配置读不进来时什么都不会执行，不会有副作用。")
            print()
            return 2

    return wrapper


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
