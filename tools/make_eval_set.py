"""第 ⓪ 层 · 从真实文档里产出评估集

【这个工具解决什么】

S1 的核心工作是人来标注 20 条真值。这是整套系统里唯一不能外包给 AI 的一步 ——
让模型替你标真值，你得到的不是标准，是回音。

但"不能外包判断"不等于"不能省事"。这个工具把人要做的事压到最小：

    机器负责：挑样本、排版、校验格式、检查分类是否均衡
    人只负责：看一眼，写两个字段

【为什么标注文件用 `p001 | 分类 | 重要性` 这种土格式】

因为它是给你的眼睛和手设计的，不是给程序设计的。
JSON 少一个逗号就全废，而且你没法一眼看出哪条还没填。
人的判断和机器的数据分开存、用 id 关联 —— 这也是一个正经的数据工程做法：
**不要让人去编辑机器格式。**

用法：
    python tools/make_eval_set.py --sample -N 20     # 抽 20 条，生成标注表
    python tools/make_eval_set.py --status           # 看填了多少
    python tools/make_eval_set.py --compile          # 编译成 eval/eval_set.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_task  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
EVAL_DIR = PROJECT_ROOT / "eval"
CANDIDATES_PATH = EVAL_DIR / "candidates.jsonl"
LABELS_PATH = EVAL_DIR / "labels.txt"
WORKSPACE_PATH = EVAL_DIR / "worksheet.md"
LITE_PATH = EVAL_DIR / "worksheet-lite.md"
EVAL_SET_PATH = EVAL_DIR / "eval_set.jsonl"

AMBIGUOUS = {"?", "？", "x", "X"}


# ------------------------------------------------------------------ 读取素材


def load_raw_items(min_chars: int) -> list[dict]:
    if not RAW_DIR.is_dir():
        return []
    items: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if int(record.get("body_chars") or 0) < min_chars:
            continue
        record.setdefault("source", "(未知来源)")
        items.append(record)
    return items


def pick_diverse(items: list[dict], count: int, per_source_cap: int) -> list[dict]:
    """按来源轮流取样。

    【为什么不直接取前 N 条】

    如果 20 条里有 18 条来自量子位，你的评估集测的是"量子位的文风下模型表现如何"，
    而不是"这个任务做得怎么样"。这叫采样偏差，它会让你的准确率好看或难看，
    但都与真实使用无关。

    轮流取样是在评估集还很小的阶段对抗偏差最便宜的办法。
    """
    buckets: dict[str, list[dict]] = {}
    for item in items:
        buckets.setdefault(item["source"], []).append(item)
    # 同一来源内部：正文长的优先（信息量大），再按 id 稳定排序
    for bucket in buckets.values():
        bucket.sort(key=lambda x: (-int(x.get("body_chars") or 0), x["id"]))

    sources = sorted(buckets)
    picked: list[dict] = []
    cursor = 0
    guard = 0
    while len(picked) < count and guard < count * len(sources) * 4 + 100:
        guard += 1
        source = sources[cursor % len(sources)]
        cursor += 1
        bucket = buckets[source]
        if not bucket:
            continue
        if sum(1 for p in picked if p["source"] == source) >= per_source_cap:
            continue
        picked.append(bucket.pop(0))
    return picked


# ------------------------------------------------------------------ 生成标注表


def write_candidates(picked: list[dict]) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": item["id"],
                "title": item["title"],
                "url": item.get("url", ""),
                "source": item["source"],
                "published": item.get("published"),
                "body": item["body"],
                "gold": {"category": None, "importance": None},
            },
            ensure_ascii=False,
        )
        for item in picked
    ]
    CANDIDATES_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_labels(picked: list[dict], task: dict) -> None:
    categories = task.get("categories", [])
    scale = task.get("importance_scale", {})
    scale_text = "  ".join(f"{k}={v}" for k, v in sorted(scale.items(), key=lambda kv: str(kv[0])))

    header = [
        f"# 标注表 · {len(picked)} 条",
        "#",
        "# 怎么填：先看 eval/worksheet-lite.md（标题+开头，5 分钟能过一遍），",
        "#         拿不准的再去 eval/worksheet.md 搜 id 看全文。",
        "#",
        "# 格式：把每行两个竖线之间补上你的判断，例如：",
        "#     p001 | 模型与技术 | 4",
        "#",
        "# 三条规则：",
        "#   1. 先自己判断，再看模型的输出。看着模型的答案标，你得到的是回音，不是标准。",
        "#   2. 拿不准就整行写问号：  p001 | ? | ?",
        "#      不确定的样本不计入准确率，但会告诉你【你的分类体系有问题】—— 这比准确率更有价值。",
        "#   3. 重要性必须是 1-5 的整数。不要全给 3，那样这一列就没有信息量了。",
        "#",
        f"# 可选分类（只能选一个）：{'、'.join(categories)}",
        f"# 重要性标准：{scale_text}",
        "#",
        "# 编辑这个文件时不要改动 id，也不要删行 —— 删一行就等于少一条样本。",
        "# ------------------------------------------------------------------",
        "",
    ]
    rows = [f"{item['id']} |  | " for item in picked]
    LABELS_PATH.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def write_worksheet(picked: list[dict], task: dict) -> None:
    """生成一个能让你只看一个文件就标完的阅读版。"""
    categories = task.get("categories", [])
    scale = task.get("importance_scale", {})

    lines = [
        "# 标注工作纸",
        "",
        f"共 {len(picked)} 条。读完每篇，把判断写进 `eval/labels.txt`（格式见该文件顶部）。",
        "",
        "**先自己判断，再看模型。** 这句话是这份工作纸存在的全部理由 ——",
        "如果你先看了模型输出再标真值，你标出来的不是标准，是模型的回音，",
        "而用回音去评估模型，准确率永远是对的，也永远没有意义。",
        "",
        "---",
        "",
        "## 分类体系（只能选一个）",
        "",
    ]
    for name in categories:
        lines.append(f"- `{name}`")
    lines += ["", "## 重要性标准", ""]
    for key, value in sorted(scale.items(), key=lambda kv: str(kv[0]), reverse=True):
        lines.append(f"- `{key}` —— {value}")
    lines += ["", "---", ""]

    for index, item in enumerate(picked, 1):
        lines += [
            f"## {index}. `{item['id']}`　{item['title']}",
            "",
            f"- 来源：{item['source']}",
            f"- 发布：{item.get('published') or '（源未提供）'}",
            f"- 正文 {item.get('body_chars')} 字" + ("（已截断）" if item.get("truncated") else ""),
        ]
        if item.get("url"):
            lines.append(f"- 链接：{item['url']}")
        lines += ["", "```text", item["body"], "```", "", "---", ""]

    lines += [
        "## 填完之后",
        "",
        "```",
        "run.cmd compile",
        "```",
        "",
        "它会检查格式、告诉你分类分布是否均衡，然后生成 `eval/eval_set.jsonl`。",
        "",
    ]
    WORKSPACE_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_worksheet_lite(picked: list[dict], task: dict, *, lead_chars: int = 150) -> None:
    """生成"速览版"工作纸：标题 + 开头一小段。

    【为什么要有这个文件】

    全文版 `worksheet.md` 有 118KB —— 20 篇 × 3000 字。
    **认真读完需要 30 到 45 分钟。** 而它恰好卡在整个目标的唯一瓶颈上：
    没有标注就没有真值，没有真值就没有基线，没有基线后面全部无从谈起。

    问题是：**分类这件事，标题通常就够了。**

    一条资讯的标题加上开头一段，90% 的情况下足以判断它属于哪一类。
    真正需要全文的，是"摘要有没有遗漏/有没有编造"那个判断 ——
    而那个判断发生在审核队列里（`run.cmd review`），不在这一步。

    所以这里把两个判断拆到两个时间点去做：
        现在：只看标题，快速分类（5 分钟）
        以后：在审核队列里对着模型输出看全文（那时候你本来就要读）

    **一个你愿意做的 5 分钟标注，胜过一个你不会做的 30 分钟标注。**
    略微粗糙但存在的真值，比精确但不存在的真值有用得多。

    拿不准的条目，去 worksheet.md 里搜它的 id 看全文 —— 那才是全文版的用途。
    """
    categories = task.get("categories", [])
    scale = task.get("importance_scale", {})

    lines = [
        "# 标注速览",
        "",
        f"{len(picked)} 条。**分类通常看标题就够了。**",
        "拿不准的，去 `worksheet.md` 里搜这个 id 看全文。",
        "",
        "> 全文版有 118KB，认真读完要半小时。速览版的用途是让这件事**真的发生** ——",
        "> 而全文的用途是解决那少数几个你真的拿不准的条目。",
        "> 次序不要搞反：先速览过一遍，再挑出三五条去查全文。",
        "",
        "## 分类体系（只能选一个）",
        "",
    ]
    for name in categories:
        lines.append(f"- `{name}`")
    lines += ["", "## 重要性（1-5）", ""]
    for key, value in sorted(scale.items(), key=lambda kv: str(kv[0]), reverse=True):
        lines.append(f"- `{key}` —— {value}")
    lines += ["", "---", ""]

    for index, item in enumerate(picked, 1):
        lead = " ".join(str(item.get("body", "")).split())
        if len(lead) > lead_chars:
            lead = lead[:lead_chars] + "……"
        lines += [
            f"### {index}　`{item['id']}`",
            f"**{item.get('title', '')}**　·　{item.get('source', '')}　·　{item.get('body_chars', 0)} 字",
            f"> {lead}",
            "",
        ]

    lines += [
        "---",
        "",
        "## 填这里",
        "",
        "打开 `eval/labels.txt`，按**同样的顺序**填：",
        "",
        "```",
        f"{picked[0]['id'] if picked else '<id>'} | 分类 | 重要性",
        "```",
        "",
        "`labels.txt` 里那 20 行的 id 顺序和上面完全一致，所以你可以照着这个顺序一路填下去，",
        "不用来回找。拿不准的写 `?`。",
        "",
        "填完：`run.cmd labels` 看进度，`run.cmd compile` 生成评估集。",
        "",
    ]
    LITE_PATH.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ 解析标注


def parse_labels(path: Path) -> dict[str, tuple[str | None, int | None, bool]]:
    """返回 id -> (分类, 重要性, 是否标为不确定)。"""
    if not path.exists():
        return {}
    labels: dict[str, tuple[str | None, int | None, bool]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        item_id = parts[0]
        category = parts[1] if len(parts) > 1 else ""
        importance_raw = parts[2] if len(parts) > 2 else ""

        if category in AMBIGUOUS or importance_raw in AMBIGUOUS:
            labels[item_id] = (None, None, True)
            continue

        importance: int | None = None
        if importance_raw:
            try:
                importance = int(importance_raw)
            except ValueError:
                importance = None
        labels[item_id] = (category or None, importance, False)
    return labels


# ------------------------------------------------------------------ 子命令


def cmd_sample(args: argparse.Namespace) -> int:
    task = load_task()
    items = load_raw_items(args.min_chars)
    if not items:
        print()
        print("  data/raw 里没有可用的文档。")
        print("  先跑：run.cmd collect")
        print(f"  （如果抓过了，检查是不是正文都短于 {args.min_chars} 字）")
        print()
        return 2

    picked = pick_diverse(items, args.count, args.per_source_cap)
    write_candidates(picked)
    write_labels(picked, task)
    write_worksheet(picked, task)
    write_worksheet_lite(picked, task)

    distribution = Counter(item["source"] for item in picked)
    lengths = sorted(int(item.get("body_chars") or 0) for item in picked)

    print()
    print("=" * 68)
    print(f"  已抽取 {len(picked)} 条样本")
    print("=" * 68)
    print(f"  来源分布      " + "，".join(f"{k} {v} 条" for k, v in sorted(distribution.items())))
    print(f"  正文长度      最短 {lengths[0]} 字，中位 {lengths[len(lengths) // 2]} 字，最长 {lengths[-1]} 字")
    print(f"  库里可用总数  {len(items)} 条")
    print()
    print("  生成了四个文件：")
    print(f"    eval/worksheet-lite.md   <- 【先看这个】标题+开头，5 分钟能过一遍")
    print(f"    eval/labels.txt          <- 【然后填这个】格式：p001 | 分类 | 重要性")
    print(f"    eval/worksheet.md        <- 全文版，拿不准的条目去这里搜 id")
    print(f"    eval/candidates.jsonl    <- 机器格式，不要手工编辑")
    print()
    print("  下一步：")
    print("    1. 打开 eval/worksheet-lite.md，从头扫到尾")
    print("    2. 每看一条，先在脑子里给出分类和重要性，然后写进 eval/labels.txt")
    print("    3. 挑出你真的拿不准的三五条，去 worksheet.md 里搜 id 看全文")
    print("    4. 拿不准的写问号，不要猜")
    print("    5. 填完跑：run.cmd compile")
    print()
    print("  这是整套系统里唯一不能交给 AI 的一步。")
    print("  速览版的存在是为了让它【真的发生】—— 一个你愿意做的 5 分钟标注，")
    print("  胜过一个你不会做的 30 分钟标注。")
    print()
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not CANDIDATES_PATH.exists():
        print("\n  还没有标注任务。先跑：run.cmd sample -N 20\n")
        return 2

    candidates = [
        json.loads(line)
        for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = parse_labels(LABELS_PATH)

    filled = ambiguous = 0
    for item in candidates:
        category, importance, is_ambiguous = labels.get(item["id"], (None, None, False))
        if is_ambiguous:
            ambiguous += 1
        elif category and importance:
            filled += 1

    total = len(candidates)
    remaining = total - filled - ambiguous
    bar_width = 40
    done_ratio = (filled + ambiguous) / total if total else 0
    bar = "#" * int(bar_width * done_ratio) + "." * (bar_width - int(bar_width * done_ratio))

    print()
    print("=" * 68)
    print("  标注进度")
    print("=" * 68)
    print(f"  [{bar}]  {done_ratio * 100:.0f}%")
    print()
    print(f"  总数        {total}")
    print(f"  已标注      {filled}")
    print(f"  标记不确定  {ambiguous}")
    print(f"  还没填      {remaining}")
    print()
    if remaining:
        missing = [
            item["id"]
            for item in candidates
            if labels.get(item["id"], (None, None, False)) == (None, None, False)
        ]
        print(f"  待填：{' '.join(missing[:24])}{' ...' if len(missing) > 24 else ''}")
        print()
    else:
        print("  填完了。跑：run.cmd compile")
        print()
    return 0


def cmd_wizard(args: argparse.Namespace) -> int:
    """标注向导：一条一条问，你只需要按数字。

    【为什么要有这个】

    原来的标注流程要求你做这些事：
        跑到某个文件夹 → 用记事本打开一个 .md → 在一堆 # 和 * 里找正文
        → 打开另一个文件 → 在 20 个十六进制编号之间来回对照 → 保存
        → 而且不能存错编码

    这一串全是【我】制造出来的麻烦，和技术本身毫无关系。
    一个不懂技术的人卡在这些步骤上，是完全正常的。

    向导把这些全部去掉：
        打开一个窗口 → 它显示第 1 条 → 你按一个数字 → 它显示第 2 条 → ……

    设计上有意的几条：
      · 只按数字，不用打中文（输入法切换是真实存在的摩擦）
      · 每答一条立刻存盘 —— 你随时关掉窗口，已经答完的不会丢
      · 再打开时自动从没答完的地方继续
      · 想不起来这条讲什么，按 m 看全文
      · 输入什么都不对的时候，它再说一遍怎么输，而不是报错退出
    """
    task = load_task()
    categories = task.get("categories", [])
    scale = task.get("importance_scale", {})

    if not CANDIDATES_PATH.exists():
        print("\n  还没有可标注的样本。先跑：run.cmd sample -N 20\n")
        return 2

    candidates = [
        json.loads(line)
        for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    order = [item["id"] for item in candidates]
    by_id = {item["id"]: item for item in candidates}
    answers = parse_labels(LABELS_PATH)

    def is_done(item_id: str) -> bool:
        category, importance, ambiguous = answers.get(item_id, (None, None, False))
        return bool(ambiguous or (category and importance))

    remaining = [item_id for item_id in order if not is_done(item_id)]
    done = len(order) - len(remaining)

    print()
    print("=" * 66)
    print(f"  标注向导　共 {len(order)} 条，已填 {done} 条，还剩 {len(remaining)} 条")
    print("=" * 66)
    if not remaining:
        print()
        print("  全部填完了。跑：run.cmd compile")
        print()
        return 0

    print()
    print("  一次问一条。看懂了就按一个数字，不用打中文。")
    print("  随时按 Ctrl+C 退出 —— 已经答完的会立刻存盘，下次接着来。")
    print()

    def save() -> None:
        _rewrite_labels(order, answers)

    def show(item_id: str, index: int) -> None:
        item = by_id[item_id]
        body = " ".join(str(item.get("body", "")).split())
        # candidates.jsonl 里没有 body_chars 字段，直接数一遍就行。
        # 第一版用了 item.get("body_chars", 0)，于是每条都显示"0 字"。
        chars = len(body)
        print()
        print("-" * 66)
        print(f"  【第 {index} 条 / 还剩 {len(remaining) - index + 1} 条】")
        print()
        print(f"  {item.get('title', '')}")
        print(f"  来源：{item.get('source', '')}　·　{chars} 字")
        print()
        # 不做自己折行 —— 交给终端去折。
        # 第一版按"显示宽度"折行，结果把英文单词从中间劈开了
        # （"...Tsinghua Un / iversity..."），读起来非常难受。
        # 终端自己会按空格折，比我手写的规则好得多。
        print(f"  {body[:400]}")
        if chars > 400:
            print()
            print("  （开头只显示这么多。想看全文就按 m）")

    def ask_number(prompt: str, valid: range | list[int], hint: str) -> int | str | None:
        """问一个数字。返回 None 表示"拿不准"，返回 'quit' 表示要退出。"""
        while True:
            try:
                raw = input(prompt).strip()
            except EOFError:
                return "quit"
            if raw.lower() == "q":
                return "quit"
            if raw == "":
                print(f"  （{hint}）")
                continue
            if raw in ("0", "?"):
                return None
            if raw.isdigit() and int(raw) in list(valid):
                return int(raw)
            print(f"  （看不懂 {raw!r}。{hint}）")

    try:
        for index, item_id in enumerate(remaining, 1):
            item = by_id[item_id]
            show(item_id, index)

            # ---- 分类
            print()
            print("  它属于哪一类？")
            for number, name in enumerate(categories, 1):
                print(f"    {number}. {name}")
            print("    0. 拿不准（记为 ?，不计入准确率）")
            print()
            choice = ask_number(
                "  请输入数字后回车：",
                range(1, len(categories) + 1),
                f"请输入 1 到 {len(categories)} 之间的数字，或 0 表示拿不准",
            )
            if choice == "quit":
                raise KeyboardInterrupt
            category = None if choice is None else categories[choice - 1]

            # ---- 重要性
            importance = None
            if category is not None:
                print()
                print("  有多重要？")
                for key, value in sorted(scale.items(), key=lambda kv: str(kv[0]), reverse=True):
                    print(f"    {key}. {value}")
                print()
                pick = ask_number(
                    "  请输入 1-5 后回车：",
                    range(1, 6),
                    "请输入 1 到 5 之间的数字",
                )
                if pick == "quit":
                    raise KeyboardInterrupt
                importance = None if pick is None else int(pick)

            answers[item_id] = (category, importance, category is None)
            save()
            mark = "?" if category is None else f"{category} / {importance}"
            print()
            print(f"  ── 已记录：{mark}　（已存盘）")

    except (KeyboardInterrupt, EOFError):
        print()
        print()
        print("  停在这里了。已经答完的都存好了，下次跑 run.cmd wizard 会接着问。")
        save()

    answered = sum(1 for item_id in order if is_done(item_id))
    print()
    print("=" * 66)
    print(f"  进度 {answered}/{len(order)}")
    print("=" * 66)
    if answered >= len(order):
        print()
        print("  全部填完了。下一步：")
        print("    run.cmd compile")
    else:
        print()
        print("  还没填完。接着填：run.cmd wizard")
    print()
    return 0


def _rewrite_labels(order: list[str], answers: dict) -> None:
    """把 answers 写回 labels.txt，保持原有的顺序和表头。"""
    header: list[str] = []
    if LABELS_PATH.exists():
        for line in LABELS_PATH.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                header.append(line)
            elif "|" not in line:
                header.append(line)
    if not header:
        header = [f"# 标注表 · {len(order)} 条", "# 格式：<id> | 分类 | 重要性", ""]

    rows = []
    for item_id in order:
        category, importance, ambiguous = answers.get(item_id, (None, None, False))
        if ambiguous:
            rows.append(f"{item_id} | ? | ?")
        elif category and importance:
            rows.append(f"{item_id} | {category} | {importance}")
        else:
            rows.append(f"{item_id} |  | ")

    # 表头末尾补一个空行，再跟数据
    while header and not header[-1].strip():
        header.pop()
    LABELS_PATH.write_text("\n".join(header + [""] + rows) + "\n", encoding="utf-8")


def cmd_compile(args: argparse.Namespace) -> int:
    task = load_task()
    categories = task.get("categories", [])

    if not CANDIDATES_PATH.exists():
        print("\n  找不到 eval/candidates.jsonl。先跑：run.cmd sample -N 20\n")
        return 2

    candidates = [
        json.loads(line)
        for line in CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    labels = parse_labels(LABELS_PATH)

    if not labels:
        print("\n  eval/labels.txt 是空的或不存在。先跑：run.cmd sample -N 20\n")
        return 2

    records = []
    problems: list[str] = []
    ambiguous_ids: list[str] = []
    unfilled_ids: list[str] = []

    for item in candidates:
        item_id = item["id"]
        category, importance, is_ambiguous = labels.get(item_id, (None, None, False))

        if is_ambiguous:
            ambiguous_ids.append(item_id)
            gold = {"category": None, "importance": None, "ambiguous": True}
        elif not category or importance is None:
            unfilled_ids.append(item_id)
            continue
        else:
            if category not in categories:
                problems.append(f"{item_id}: 分类 {category!r} 不在允许列表里")
                continue
            if not 1 <= importance <= 5:
                problems.append(f"{item_id}: 重要性 {importance} 不在 1-5 之间")
                continue
            gold = {"category": category, "importance": importance}

        records.append(
            {
                "id": item_id,
                "title": item["title"],
                "source": item["source"],
                "published": item.get("published"),
                "url": item.get("url", ""),
                "body": item["body"],
                "gold": gold,
            }
        )

    print()
    print("=" * 68)
    print("  编译检查")
    print("=" * 68)

    if problems:
        print()
        print(f"  有 {len(problems)} 处格式问题：")
        for problem in problems:
            print(f"    ! {problem}")
        print()
        print("  改完 eval/labels.txt 再跑一次。")
        print()
        return 1

    if unfilled_ids:
        print()
        print(f"  还有 {len(unfilled_ids)} 条没填：{' '.join(unfilled_ids[:24])}")
        if not args.allow_partial:
            print()
            print("  要么填完它们，要么加 --allow-partial 只编译已填的部分。")
            print("  （加 --allow-partial 也可以，但样本越少，准确率的误差越大。）")
            print()
            return 1
        print("  按 --allow-partial 处理，只编译已填的。")

    if not records:
        print("\n  没有任何可编译的样本。\n")
        return 1

    # ---------------- 质量检查：这些小字比编译本身更重要 ----------------

    warnings: list[str] = []
    scored = [r for r in records if not r["gold"].get("ambiguous")]

    if len(records) < 10:
        warnings.append(
            f"样本只有 {len(records)} 条。少于 10 条时，准确率的一个样本就值 10 个百分点，"
            f"数字会非常不可靠。"
        )

    if scored:
        category_counts = Counter(r["gold"]["category"] for r in scored)
        top_category, top_count = category_counts.most_common(1)[0]
        share = top_count / len(scored)
        if share > 0.6:
            warnings.append(
                f"分类严重不均衡：{top_count}/{len(scored)} 条都是「{top_category}」（{share:.0%}）。\n"
                f"      这种评估集测不出区分度 —— 一个永远回答「{top_category}」的模型也能拿到 {share:.0%}。\n"
                f"      补几条其他分类的样本，或者把评估集的目标改成'能不能挑出来'。"
            )
        unused = [c for c in categories if c not in category_counts]
        if unused:
            warnings.append(f"从未用到的分类：{'、'.join(unused)}（不一定有问题，但值得知道）")

        importance_counts = Counter(r["gold"]["importance"] for r in scored)
        if len(importance_counts) == 1:
            only = importance_counts.most_common(1)[0][0]
            warnings.append(
                f"重要性全都是 {only}。这一列没有信息量了，等于没标。\n"
                f"      回到 worksheet.md 重新看一眼：真的每条都一样重要吗？"
            )

    short = [r for r in records if len(r["body"]) < 300]
    if short:
        warnings.append(f"{len(short)} 条正文短于 300 字，摘要任务在它们上面测不出什么。")

    # ---------------- 写入 ----------------

    if EVAL_SET_PATH.exists() and not args.force:
        existing = EVAL_SET_PATH.read_text(encoding="utf-8")
        still_placeholder = "_placeholder" in existing
        if not still_placeholder:
            print()
            print("  eval/eval_set.jsonl 已经存在，而且看起来是你自己标过的数据。")
            print("  直接覆盖会丢掉它。确认要覆盖就加 --force。")
            print()
            return 1

    EVAL_SET_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )

    print()
    print(f"  已写入 eval/eval_set.jsonl")
    print(f"    可评分样本  {len(scored)} 条")
    print(f"    标记不确定  {len(ambiguous_ids)} 条" + (f"（{' '.join(ambiguous_ids[:12])}）" if ambiguous_ids else ""))
    if scored:
        print(f"    分类分布    " + "，".join(f"{k} {v}" for k, v in category_counts.most_common()))

    if warnings:
        print()
        print("  " + "-" * 64)
        print("  值得先处理的问题：")
        for warning in warnings:
            print(f"    * {warning}")
        print()
        print("  这些不是错误，不影响使用。但每一条都会削弱你后面那个数字的可信度。")
    else:
        print()
        print("  评估集看起来是健康的。")

    print()
    print("  下一步：先填 config/providers.json 的单价，然后")
    print("    run.cmd eval -Limit 3      # 花几分钱验证链路")
    print("    run.cmd eval               # 跑出你的基线")
    print()
    return 0


# ------------------------------------------------------------------ 入口


def main() -> int:
    parser = argparse.ArgumentParser(description="从真实文档生成评估集")
    parser.add_argument("--sample", "-Sample", action="store_true", help="抽样本，生成标注表")
    parser.add_argument("--compile", "-Compile", action="store_true", help="把标注编译成评估集")
    parser.add_argument("--status", "-Status", action="store_true", help="查看标注进度")
    parser.add_argument("--wizard", "-Wizard", action="store_true",
                        help="一条一条问你的标注向导（推荐：不用打中文，不用对编号）")
    parser.add_argument("-N", "--count", type=int, default=20, help="抽多少条（默认 20）")
    parser.add_argument("--min-chars", type=int, default=300, help="正文短于这个字数的不要")
    parser.add_argument("--per-source-cap", type=int, default=6, help="同一来源最多取几条")
    parser.add_argument("--allow-partial", action="store_true", help="允许只编译已填的部分")
    parser.add_argument("--force", action="store_true", help="覆盖已有的评估集")
    args = parser.parse_args()

    if args.wizard:
        return cmd_wizard(args)
    if args.compile:
        return cmd_compile(args)
    if args.status:
        return cmd_status(args)
    return cmd_sample(args)


if __name__ == "__main__":
    raise SystemExit(main())
