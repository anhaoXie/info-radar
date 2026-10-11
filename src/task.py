"""第 ⓞ 层 · 任务定义 → 消息 与 校验

把 config/task.json 变成两样东西：

    1. 发给模型的消息      build_messages()
    2. 对模型输出的结构校验  validate()

【为什么校验必须和提示词住在同一个文件里】

因为"输出格式是否合法"本身就是任务的判据之一。
判据和提示词分开写，它们迟早会不一致 ——
然后你会开始怀疑模型，而问题出在你自己这边。
"""

from __future__ import annotations

import json
from typing import Any

from .config import load_task

REQUIRED_FIELDS = ("title", "summary", "category", "importance", "entities", "so_what", "evidence")


# ------------------------------------------------------------------ 解析


def _extract_balanced(text: str, opener: str = "{", closer: str = "}") -> str | None:
    """从一段文本里取出第一个括号配平的片段。

    要正确处理字符串里的括号和转义，否则遇到 {"a": "}"} 这种内容就会切错。
    """
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_json(text: str) -> tuple[dict | None, str | None]:
    """尽力从模型输出里取出一个 JSON 对象。

    为什么要"尽力"，而不是直接 json.loads：
    没有任何一家厂商的 JSON 模式是 100% 可靠的。它们仍然会偶尔包上 ``` 围栏，
    或者先来一句"好的，以下是结果："。这不代表模型坏了，
    代表你必须防御性解析 —— 这是使用生成式模型时的常态，不是意外。
    """
    if not text or not text.strip():
        return None, "输出为空"

    candidate = text.strip()

    # 去掉 markdown 代码围栏
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()

    try:
        parsed = json.loads(candidate)
        return (parsed, None) if isinstance(parsed, dict) else (None, f"顶层不是对象，而是 {type(parsed).__name__}")
    except json.JSONDecodeError:
        pass

    # 退一步：从文本里抠出第一个配平的 {...}
    block = _extract_balanced(candidate)
    if block:
        try:
            parsed = json.loads(block)
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError as exc:
            return None, f"括号配平了但 JSON 非法：{exc}"

    return None, f"找不到合法的 JSON 对象。原始输出前 200 字：{candidate[:200]}"


# ------------------------------------------------------------------ 校验


def validate(task: dict, obj: Any) -> tuple[list[str], list[str]]:
    """结构校验。返回 (硬伤, 提醒)。

    硬伤 = 格式不合法，算作这一条失败。
    提醒 = 值得注意但不判失败（比如摘要长了几个字）。
    把这两者分开，你的"格式合规率"才是一个有意义的数字。
    """
    problems: list[str] = []
    warnings: list[str] = []

    if not isinstance(obj, dict):
        return ["输出不是一个 JSON 对象"], []

    for field_name in REQUIRED_FIELDS:
        if field_name not in obj:
            problems.append(f"缺字段 {field_name}")
    if problems:
        return problems, warnings

    categories = task.get("categories", [])
    category = obj.get("category")
    if category not in categories:
        problems.append(f"category 不在允许列表里：{category!r}")

    importance = obj.get("importance")
    if not isinstance(importance, int) or isinstance(importance, bool) or not 1 <= importance <= 5:
        problems.append(f"importance 必须是 1-5 的整数，实际是 {importance!r}")

    if not isinstance(obj.get("title"), str) or not obj["title"].strip():
        problems.append("title 为空")

    if not isinstance(obj.get("summary"), str) or not obj["summary"].strip():
        problems.append("summary 为空")

    entities = obj.get("entities")
    if not isinstance(entities, list):
        problems.append("entities 必须是列表")
    elif len(entities) > 5:
        warnings.append(f"entities 有 {len(entities)} 个，建议不超过 5 个")

    evidence = obj.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        problems.append("evidence 必须是至少含一条的列表（它是防幻觉的锚点，不能省）")
    elif len(evidence) > 3:
        warnings.append(f"evidence 有 {len(evidence)} 条，建议不超过 3 条")

    if len(str(obj.get("title", ""))) > 30:
        warnings.append(f"title 长 {len(str(obj['title']))} 字，超过 30 字")
    if len(str(obj.get("summary", ""))) > 120:
        warnings.append(f"summary 长 {len(str(obj['summary']))} 字，超过 120 字")
    if len(str(obj.get("so_what", ""))) > 50:
        warnings.append(f"so_what 长 {len(str(obj['so_what']))} 字，超过 50 字")

    return problems, warnings


# ------------------------------------------------------------------ 组装


def _render_scale(scale: dict) -> str:
    return "\n".join(f"{key} = {value}" for key, value in sorted(scale.items(), key=lambda kv: str(kv[0])))


def build_messages(task: dict, doc: dict) -> list[dict]:
    """把「任务定义 + 一篇文档」渲染成消息序列。

    注意 few_shot 的位置：示例必须放在真正的提问【之前】，
    且成对出现（user / assistant）。
    config/task.json 里默认是空数组 —— 这是故意的：
    S1 的基线必须是最笨的版本，否则你无法测量"加示例"到底起了多大作用。
    """
    prompt = task.get("prompt", {})
    template = prompt.get("user_template", "{body}")
    rendered = template.format(
        title=doc.get("title", ""),
        source=doc.get("source", ""),
        published=doc.get("published", ""),
        body=doc.get("body", ""),
        categories="、".join(task.get("categories", [])),
        importance_scale=_render_scale(task.get("importance_scale", {})),
    )

    messages: list[dict] = [{"role": "system", "content": prompt.get("system", "")}]

    for example in prompt.get("few_shot", []):
        ex_doc = example.get("input", {})
        messages.append({"role": "user", "content": template.format(
            title=ex_doc.get("title", ""),
            source=ex_doc.get("source", ""),
            published=ex_doc.get("published", ""),
            body=ex_doc.get("body", ""),
            categories="、".join(task.get("categories", [])),
            importance_scale=_render_scale(task.get("importance_scale", {})),
        )})
        messages.append({
            "role": "assistant",
            "content": json.dumps(example.get("output", {}), ensure_ascii=False),
        })

    messages.append({"role": "user", "content": rendered})
    return messages


def load() -> dict:
    return load_task()
