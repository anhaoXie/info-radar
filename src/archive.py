"""第 ④ 层 · 分发与审核：人在哪一步介入，以及审核结果去了哪

【这一层解决两个问题，第二个比第一个重要得多】

  1. 把 ③ 层的机器格式产出，变成人能用眼睛扫的东西（日报）。
  2. 让人的每一次审核，**回流成评估集的真值样本**。

第 2 条是整套系统里唯一一处"它会自己变强"的地方。

【为什么未审核的产出不能流向"会被当真"的地方】

归档到本地随便怎么放都行 —— 错了你自己知道。
但发邮件、写进共享文档、推到群里，就变成"别人以为这是真的"。
所以本层默认产出两份东西：

    out/digests/<stamp>.md      日报（标注着"未审核"）
    out/review/<stamp>.txt      审核表（人填）

人填完，再生成终版。**"未审核"这三个字必须真的出现在文件里** ——
一个只在文档里约定的规矩，会在你最忙的那一周被跳过。

【闭环：这就是评估集长大的方式】

S1 你手工标了 20 条，很累。
但如果每天审核日报里顺手改几条，一个月后评估集就是几百条 ——
而且这些样本来自**真实流量**，比手工挑的 20 条更接近真实分布。

于是"我对系统的信任"就从感觉变成了统计。

【一个必须讲清楚的陷阱：不要让评估集变成模型的回音】

审核表里留空 = "我读了，我认可"。这是有效的人工标注。
但如果你只是扫一眼就留空，你标出来的不是标准，是模型的复述 ——
用它去评估模型，准确率永远好看，也永远没意义。

所以这里做了一件事：从审核回流来的样本会带上 `label_origin: "review"`。
将来你可以分别统计"手工标注样本上的准确率"和"审核回流样本上的准确率"。
如果后者明显更高，那说明的不是模型好，是**你审得不够认真**。

用法：
    python src/archive.py                          # 用最新一轮结果生成日报 + 审核表
    python src/archive.py --list                   # 看历史产出
    python src/archive.py --apply 20261011-210000  # 审核完，生成终版并回流评估集
    python src/archive.py --stats                  # 看审核回流累计了多少真值
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_task, read_text_smart, main_guard  # noqa: E402
from src.secrets import PROJECT_ROOT as ROOT  # noqa: E402

RESULTS_DIR = ROOT / "out" / "results"
DIGEST_DIR = ROOT / "out" / "digests"
REVIEW_DIR = ROOT / "out" / "review"
RAW_DIR = ROOT / "data" / "raw"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"

AMBIGUOUS = {"?", "？"}
REJECT_MARKS = {"x", "X", "×", "✗"}


# ------------------------------------------------------------------ 读取


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def list_result_sets() -> list[Path]:
    if not RESULTS_DIR.is_dir():
        return []
    return sorted(RESULTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


def resolve_result_set(stamp: str | None = None) -> Path | None:
    """找出要处理的那一轮结果。给 stamp 就找它，否则用最新的一轮。"""
    sets = list_result_sets()
    if not sets:
        return None
    if not stamp:
        return sets[0]
    for path in sets:
        if path.stem == stamp or path.stem.startswith(stamp):
            return path
    return None


def load_result_set(path: Path) -> tuple[list[dict], dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    summary = _load_json(path.with_suffix(".summary.json"), {})
    return records, summary


# ------------------------------------------------------------------ 日报


def _category_order(task_cfg: dict) -> list[str]:
    return list(task_cfg.get("categories", []))


def render_digest(
    records: list[dict],
    summary: dict,
    *,
    stamp: str,
    task_cfg: dict,
    reviewed: dict | None = None,
    final: bool = False,
) -> str:
    """把机器格式的结果渲染成人能扫的日报。

    reviewed 传进来时，用人体审核后的分类/重要性覆盖模型的判断 ——
    这是"人覆盖机器"的唯一入口，也是为什么审核必须有实际效果：
    **一个改了却不起作用的审核，没人会做第二次。**
    """
    reviewed = reviewed or {}
    categories = _category_order(task_cfg)
    scale = task_cfg.get("importance_scale", {})

    rows = []
    for record in records:
        item_id = record.get("id", "")
        decision = reviewed.get(item_id)
        if decision is not None and decision.get("included") is False:
            continue

        output = record.get("output") or {}
        category = output.get("category")
        importance = output.get("importance")
        corrected = False
        if decision:
            if decision.get("category"):
                if decision["category"] != category:
                    corrected = True
                category = decision["category"]
            if decision.get("importance") is not None:
                if decision["importance"] != importance:
                    corrected = True
                importance = decision["importance"]

        rows.append(
            {
                "id": item_id,
                "title": record.get("title") or output.get("title") or "(无标题)",
                "source": record.get("source", ""),
                "status": record.get("status", ""),
                "output": output,
                "category": category if category in categories else "其他",
                "importance": importance if isinstance(importance, int) else 0,
                "corrected": corrected,
                "ambiguous": bool(decision and decision.get("ambiguous")),
            }
        )

    rows.sort(key=lambda r: (-r["importance"], r["category"], r["title"]))

    # ---- 产出不完整的那些，必须在【头部】就点明
    #
    # 失败的条目其实【会】出现在日报里，排在「其他」组、标着 `产出不完整`。
    # 所以这里不能说"它们没有出现在下面" —— 那是假的。
    #
    # 但只在正文里标一个记号是不够的：你扫头部时看到"收录 5 条"，
    # 会以为今天是 5 条新闻，而其中 2 条其实是失败的空壳。
    #
    # **"今天新闻少"和"两条没处理成功"在头部看起来完全一样。**
    # 开头这一句就是把它们分开的那一句。
    failed = [r for r in records if str(r.get("status") or "ok") != "ok"]
    failure_lines: list[str] = []
    if failed:
        reason_label = {"format_error": "产出不合规", "call_failed": "调用失败"}
        counts: dict[str, int] = {}
        for record in failed:
            key = str(record.get("status"))
            counts[key] = counts.get(key, 0) + 1
        detail = "，".join(f"{reason_label.get(k, k)} {v} 篇" for k, v in counts.items())
        failure_lines = [
            f"> **其中 {len(failed)} 篇产出不完整**（{detail}）。",
            "> 它们排在下面「其他」里、标着 `产出不完整`，内容不可用。",
            "> 下次跑流水线加 `--retry-failed` 可以只重试这些。",
            "",
        ]

    heads = [
        f"# 行业情报日报 · {stamp[:8]}",
        "",
        f"> 生成时间　{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"> 收录条数　{len(rows)}",
    ]
    heads += failure_lines
    heads += [
        f"> 模型　　　{summary.get('provider', '?')} / {summary.get('model', '?')}",
        f"> 本轮成本　{summary.get('total_cost_cny', 0)} 元",
        "",
    ]
    if final:
        heads += [
            "> **已审核（终版）** —— 分类与重要性来自人工确认。",
            "",
        ]
    else:
        heads += [
            "> **未审核** —— 这份产出还没有人看过。",
            "> 不要把它直接发出去：跑 `run.cmd review --apply " + stamp + "` 生成终版。",
            "",
        ]

    heads += ["---", ""]

    if not rows:
        heads += ["（这一轮没有可收录的条目。）", ""]
        return "\n".join(heads)

    # 分组顺序 = 配置里的分类顺序 + 兜底的"其他"。
    # 必须去重：config/task.json 的 categories 本身就以「其他」结尾，
    # 直接 categories + ["其他"] 会让这一组被渲染两遍 —— 日报里每条都出现两次。
    # 这个 bug 只有真的打开生成的日报才会看见，测试如果不查"出现几次"就抓不到。
    order = list(dict.fromkeys(list(categories) + ["其他"]))

    for category in order:
        group = [r for r in rows if r["category"] == category]
        if not group:
            continue
        heads += [f"## {category}", ""]
        for row in group:
            output = row["output"]
            marks = []
            if row["corrected"]:
                marks.append("人工修正")
            if row["ambiguous"]:
                marks.append("分类存疑")
            if row["status"] != "ok":
                marks.append("产出不完整")
            suffix = f"　`{'·'.join(marks)}`" if marks else ""

            heads.append(f"### [{row['importance']}] {row['title']}{suffix}")
            heads.append("")
            if output.get("summary"):
                heads.append(f"**摘要**　{output['summary']}")
                heads.append("")
            if output.get("so_what"):
                heads.append(f"**影响**　{output['so_what']}")
                heads.append("")
            if output.get("entities"):
                heads.append(f"**实体**　{'、'.join(str(e) for e in output['entities'][:5])}")
                heads.append("")
            for piece in (output.get("evidence") or [])[:2]:
                heads.append(f"> {piece}")
            if output.get("evidence"):
                heads.append("")

            meta_bits = [b for b in [row["source"]] if b]
            meta_bits.append(f"`{row['id']}`")
            heads.append("　".join(meta_bits))
            heads.append("")
        heads += ["---", ""]

    if scale:
        heads += ["## 重要性标准", ""]
        for key, value in sorted(scale.items(), key=lambda kv: str(kv[0]), reverse=True):
            heads.append(f"- `{key}` —— {value}")
        heads.append("")

    return "\n".join(heads)


# ------------------------------------------------------------------ 审核表


def render_review(records: list[dict], summary: dict, *, stamp: str, task_cfg: dict) -> str:
    """生成审核表。

    格式是给手写的，不是给程序读的 —— 和 S1 的 labels.txt 同一个道理：
    不要让人去编辑机器格式。JSON 少一个逗号就全废，
    而且你没法一眼看出哪条还没看。
    """
    categories = _category_order(task_cfg)
    usable = [r for r in records if r.get("status") == "ok"]
    broken = [r for r in records if r.get("status") != "ok"]

    lines = [
        f"# 审核表 · {stamp}",
        "",
        f"共 {len(records)} 条（其中 {len(broken)} 条产出不完整）。",
        "",
        "## 怎么填",
        "",
        "每一行一条，四列。**收录列留空就是通过**；",
        "分类和重要性两列已经预填了模型的判断，**你不改就是认可**。",
        "",
        "     <id> | 收录 | 分类 | 重要性",
        "",
        "注意上面这行是【缩进】的，不是用 ``` 包起来的。",
        "这不是排版偏好：解析器靠 ``` 来划分数据区，",
        "把格式示例放进围栏里，它会被当成一条真数据读进去 ——",
        "于是审核表里凭空多出一个叫 `<id>` 的条目。",
        "",
        "| 列 | 默认 | 想改就写 |",
        "|---|---|---|",
        "| 收录 | 留空 = 通过，收进日报 | `x` = 不收录 |",
        "| 分类 | 预填了模型的判断，不改 = 认可 | 改成正确的分类名 |",
        "| 重要性 | 预填了模型的打分，不改 = 认可 | 写 `1`–`5` |",
        "",
        "**「不改」等于「我读了，我认可」。** 这句话是真的承诺，不是免责声明。",
        "如果你只是扫一眼就不改，你标出来的不是标准，是模型的复述 ——",
        "用它去评估模型，准确率永远好看，也永远没有意义。",
        "",
        "## 填完之后",
        "",
        "```",
        f"run.cmd review --apply {stamp}",
        "```",
        "",
        "它会生成终版日报（排除你不要的、用你改后的分类和重要性重排），",
        "并把你审过的条目**回流进评估集**。",
        "",
        f"可选分类：{'、'.join(categories)}",
        "分类实在拿不准就写 `?`，它会被标记为存疑样本。",
        "",
        "---",
        "",
    ]

    if broken:
        lines += [
            "## 产出不完整的条目",
            "",
            "这些条目模型没给出可用结果。它们的正文是好的，",
            "如果你愿意，可以直接把分类和重要性填上 —— 那就是纯人工标注。",
            "",
        ]
        for record in broken:
            output = record.get("output") or {}
            lines.append(f"- `{record.get('id')}`　{record.get('title', '')}")
            if record.get("error"):
                lines.append(f"  - 失败原因：{str(record['error']).splitlines()[0][:110]}")
            elif record.get("problems"):
                lines.append(f"  - 结构问题：{'; '.join(record['problems'])[:110]}")
        lines += ["", "---", ""]

    lines += ["## 逐条审核", "", "```text"]
    for record in records:
        output = record.get("output") or {}
        category = output.get("category") or ""
        importance = output.get("importance")
        lines.append(f"{record.get('id')} |  | {category} | {importance if importance is not None else ''}")
    lines += ["```", ""]

    return "\n".join(lines)


# ------------------------------------------------------------------ 解析审核


def parse_review(path: Path) -> dict[str, dict]:
    """解析填好的审核表。返回 id -> {included, category, importance, ambiguous}。"""
    decisions: dict[str, dict] = {}
    if not path.exists():
        return decisions

    inside_block = False
    for raw in read_text_smart(path).splitlines():
        line = raw.strip()
        if line.startswith("```"):
            inside_block = not inside_block
            continue
        if not inside_block:
            continue
        if not line or line.startswith("#") or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        item_id = parts[0]
        # 只接受"像 id"的东西。
        # 这一道不是因为上面那个格式示例的 bug 才加的 —— 它是防御性的：
        # 只要解析器靠符号划分数据区，就总会有别的东西混进那个区域，
        # 比如你顺手在审核表里写的一行备注、或者从别处粘进来的一行表格。
        # 与其相信"那个区域里只有数据"，不如在这里多问一句"这行像不像数据"。
        if not item_id or item_id.startswith("<") or any(ch.isspace() for ch in item_id):
            continue
        included_raw = parts[1] if len(parts) > 1 else ""
        category_raw = parts[2] if len(parts) > 2 else ""
        importance_raw = parts[3] if len(parts) > 3 else ""

        decision: dict = {
            "included": included_raw not in REJECT_MARKS,
            "category": None,
            "importance": None,
            "ambiguous": False,
        }

        if category_raw in AMBIGUOUS:
            decision["ambiguous"] = True
        elif category_raw:
            decision["category"] = category_raw

        if importance_raw:
            try:
                decision["importance"] = int(importance_raw)
            except ValueError:
                decision["importance"] = None

        decisions[item_id] = decision

    return decisions


# ------------------------------------------------------------------ 回流评估集


def load_eval_set_ids() -> set[str]:
    if not EVAL_SET_PATH.exists():
        return set()
    ids = set()
    for line in read_text_smart(EVAL_SET_PATH).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            ids.add(str(json.loads(line).get("id")))
        except json.JSONDecodeError:
            continue
    return ids


def append_to_eval_set(rows: list[dict]) -> int:
    """把审核结果追加进评估集。已存在的 id 跳过。

    【为什么是"重写整个文件"而不是"追加一行"】

    追加看起来更省事，但它有一个真实的坏结局：
    `eval_set.jsonl` 是【明确希望用户手工编辑】的文件。你在记事本里
    改一下、保存 —— 中文 Windows 的记事本会把它存成 GBK。
    这时再用 utf-8 追加，文件里就同时有了两种编码的字节。
    之后无论用哪种编码去读，总有一半是乱码，而且很难看出是从哪一行开始坏的。

    所以这里改成：整份读进来（宽容地读），加上新行，再整份用 utf-8 写回去。
    顺便把编码统一了 —— 用户无意中造成的 GBK 文件，会在这一步被顺回 utf-8。
    """
    if not rows:
        return 0
    EVAL_SET_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = load_eval_set_ids()
    fresh = [r for r in rows if str(r.get("id")) not in existing]
    if not fresh:
        return 0

    # 保留原有的所有内容（含注释行），只在末尾追加
    old_lines: list[str] = []
    if EVAL_SET_PATH.exists():
        old_lines = [
            line for line in read_text_smart(EVAL_SET_PATH).splitlines() if line.strip()
        ]
    new_lines = [json.dumps(row, ensure_ascii=False) for row in fresh]
    EVAL_SET_PATH.write_text("\n".join(old_lines + new_lines) + "\n", encoding="utf-8")
    return len(fresh)


def read_raw_body(item_id: str) -> dict | None:
    path = RAW_DIR / f"{item_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ------------------------------------------------------------------ 主流程


def cmd_digest(stamp: str | None, quiet: bool) -> int:
    path = resolve_result_set(stamp)
    if path is None:
        print()
        print("  out/results/ 里还没有任何产出。")
        print("  先跑：run.cmd pipeline --stub   （不需要密钥）")
        print("  或者：run.cmd pipeline          （要密钥）")
        print()
        return 2

    records, summary = load_result_set(path)
    if not records:
        print(f"\n  {path.name} 是空的。\n")
        return 2

    task_cfg = load_task()
    stamp_value = path.stem

    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    digest_path = DIGEST_DIR / f"{stamp_value}.md"
    digest_path.write_text(
        render_digest(records, summary, stamp=stamp_value, task_cfg=task_cfg),
        encoding="utf-8",
    )
    review_path = REVIEW_DIR / f"{stamp_value}.txt"
    review_path.write_text(
        render_review(records, summary, stamp=stamp_value, task_cfg=task_cfg),
        encoding="utf-8",
    )

    ok = sum(1 for r in records if r.get("status") == "ok")
    print()
    print("=" * 68)
    print("  生成完成")
    print("=" * 68)
    print(f"  来源结果   out/results/{path.name}")
    print(f"  日报       out/digests/{digest_path.name}   （{ok} 条可用）")
    print(f"  审核表     out/review/{review_path.name}")
    print()
    print("  这份日报标注着【未审核】，不要直接发出去。")
    print()
    print("  下一步：")
    print(f"    1. 打开 out/digests/{digest_path.name} 通读一遍")
    print(f"    2. 在 out/review/{review_path.name} 里改掉不对的地方")
    print(f"    3. run.cmd review --apply {stamp_value}")
    print()
    print("  第 3 步会生成终版日报，并把你审过的条目回流进评估集。")
    print("  这是整套系统里唯一一处『它会自己变强』的地方。")
    print()
    return 0


def cmd_apply(stamp: str | None, quiet: bool) -> int:
    path = resolve_result_set(stamp)
    if path is None:
        print("\n  out/results/ 里还没有任何产出。\n")
        return 2

    stamp_value = path.stem
    review_path = REVIEW_DIR / f"{stamp_value}.txt"
    if not review_path.exists():
        print(f"\n  找不到审核表：{review_path}")
        print(f"  先跑：run.cmd digest\n")
        return 2

    decisions = parse_review(review_path)
    if not decisions:
        print(f"\n  审核表里没有读到任何条目。检查一下 out/review/{review_path.name}")
        print("  逐条审核的区块要用 ```text 包起来。\n")
        return 2

    records, summary = load_result_set(path)
    task_cfg = load_task()
    categories = _category_order(task_cfg)

    # ---------------- 统计：这一轮的审核告诉了你什么 ----------------
    rejected = [i for i, d in decisions.items() if not d["included"]]
    category_changes = []
    importance_changes = []
    ambiguous = [i for i, d in decisions.items() if d["ambiguous"]]

    by_id = {r.get("id"): r for r in records}
    for item_id, decision in decisions.items():
        record = by_id.get(item_id)
        if record is None:
            continue
        output = record.get("output") or {}
        if decision["category"] and decision["category"] != output.get("category"):
            category_changes.append(item_id)
        if decision["importance"] is not None and decision["importance"] != output.get("importance"):
            importance_changes.append(item_id)

    # ---------------- 终版日报 ----------------
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    final_path = DIGEST_DIR / f"{stamp_value}-final.md"
    final_path.write_text(
        render_digest(
            records, summary, stamp=stamp_value, task_cfg=task_cfg, reviewed=decisions, final=True
        ),
        encoding="utf-8",
    )

    # ---------------- 回流评估集 ----------------
    rows = []
    skipped_no_body = []
    for item_id, decision in decisions.items():
        if decision["ambiguous"]:
            continue
        record = by_id.get(item_id)
        if record is None:
            continue
        output = record.get("output") or {}
        category = decision["category"] or output.get("category")
        importance = decision["importance"]
        if importance is None:
            importance = output.get("importance")
        if category not in categories or not isinstance(importance, int):
            continue

        raw = read_raw_body(item_id)
        if raw is None:
            skipped_no_body.append(item_id)
            continue

        rows.append(
            {
                "id": item_id,
                "title": raw.get("title", ""),
                "source": raw.get("source", ""),
                "published": raw.get("published"),
                "url": raw.get("url", ""),
                "body": raw.get("body", ""),
                "gold": {"category": category, "importance": importance},
                # 来源标记：将来可以分开统计"手工标注"和"审核回流"上的准确率。
                # 如果后者明显更高，说明的不是模型好，是你审得不够认真。
                "label_origin": "review",
                # 「值不值得收录」和「属于哪个分类」是两个不同的判断。
                # 把它们混成一个标签，是这一层最容易犯的建模错误：
                # 一条被你标为"不收录"的资讯，它的分类依然可能是对的，
                # 所以它的分类标注依然有效 —— 但它对"相关性"这件事表达的是另一个意思。
                # 分开记下来，将来你既可以用它评分类，也可以用它评"该不该报"。
                "review_included": bool(decision["included"]),
                "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
        )

    appended = append_to_eval_set(rows)

    # ---------------- 报告 ----------------
    print()
    print("=" * 68)
    print(f"  审核已应用 · {stamp_value}")
    print("=" * 68)
    print(f"  审核条目   {len(decisions)}")
    print(f"  不收录     {len(rejected)}")
    print(f"  分类改判   {len(category_changes)}")
    print(f"  重要性改判 {len(importance_changes)}")
    print(f"  标记存疑   {len(ambiguous)}")
    print()

    judged = len(decisions) - len(ambiguous)
    if judged:
        print(f"  模型分类被你改掉了 {len(category_changes)}/{judged}"
              f" = {len(category_changes) / judged * 100:.0f}%")
        print("  这个数字就是它在真实流量上的准确率 —— 不需要额外跑一次评估。")
        print("  **前提是你真的每一条都读了。** 扫一眼就留空的话，这个数字没有意义。")
    print()
    print(f"  终版日报   out/digests/{final_path.name}")
    print(f"  回流评估集 {appended} 条新真值")
    if rejected:
        print(f"    其中 {len(rejected)} 条被你标为不收录 —— 它们的【分类】标注依然有效，")
        print("    「值不值得报」这个判断记在 review_included 字段里，两者没有混在一起。")
    if skipped_no_body:
        print(f"  跳过       {len(skipped_no_body)} 条（data/raw 里找不到原文，可能是清理过）")
    print()

    total = len(load_eval_set_ids())
    print(f"  评估集现在有 {total} 条样本。")
    if total < 40:
        print("  样本还少，准确率的波动会很大。坚持每天审核，它会自己长起来。")
    print()
    return 0


def cmd_stats() -> int:
    rows = []
    if EVAL_SET_PATH.exists():
        for line in read_text_smart(EVAL_SET_PATH).splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    from collections import Counter

    origin = Counter(str(r.get("label_origin") or "手工标注") for r in rows)
    placeholder = sum(1 for r in rows if r.get("_placeholder"))

    print()
    print("=" * 68)
    print("  评估集统计")
    print("=" * 68)
    print(f"  总样本     {len(rows)}")
    for name, count in origin.most_common():
        print(f"    {name:<10} {count}")
    if placeholder:
        print(f"  其中占位数据 {placeholder} 条（跑 run.cmd compile 会替换掉）")
    print()

    reviewed = [p for p in REVIEW_DIR.glob("*.txt")] if REVIEW_DIR.is_dir() else []
    digested = [p for p in DIGEST_DIR.glob("*.md")] if DIGEST_DIR.is_dir() else []
    finals = [p for p in digested if p.stem.endswith("-final")]
    print(f"  审核表     {len(reviewed)} 份")
    print(f"  日报       {len(digested)} 份（其中终版 {len(finals)} 份）")
    if reviewed and not finals:
        print()
        print("  有审核表但还没有终版 —— 填完跑 run.cmd review --apply <stamp>")
    print()
    return 0


def cmd_list() -> int:
    sets = list_result_sets()
    print()
    print("=" * 68)
    print("  历史产出")
    print("=" * 68)
    if not sets:
        print("  还没有任何一轮结果。跑 run.cmd pipeline --stub 先看看流程。")
        print()
        return 0
    print(f"  {'时间戳':<18}{'条数':>6}{'成本':>12}  日报 / 审核")
    print("  " + "-" * 62)
    for path in sets:
        records, summary = load_result_set(path)
        digest = (DIGEST_DIR / f"{path.stem}.md").exists()
        final = (DIGEST_DIR / f"{path.stem}-final.md").exists()
        review = (REVIEW_DIR / f"{path.stem}.txt").exists()
        marks = []
        marks.append("日报" if digest else "—")
        marks.append("终版" if final else "—")
        marks.append("审核表" if review else "—")
        print(f"  {path.stem:<18}{len(records):>6}{summary.get('total_cost_cny', 0):>12}  {' / '.join(marks)}")
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="分发与审核层")
    parser.add_argument("--digest", "-Digest", action="store_true", help="生成日报和审核表（默认动作）")
    parser.add_argument("--apply", "-Apply", default=None, nargs="?", const="", help="应用审核结果；可带时间戳")
    parser.add_argument("--list", "-List", action="store_true", help="列出历史产出")
    parser.add_argument("--stats", "-Stats", action="store_true", help="评估集与审核统计")
    parser.add_argument("--stamp", default=None, help="指定要处理的那一轮时间戳")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    stamp = args.stamp or (args.apply or None)
    if args.apply is not None:
        return cmd_apply(stamp or None, args.quiet)
    if args.list:
        return cmd_list()
    if args.stats:
        return cmd_stats()
    return cmd_digest(stamp, args.quiet)


if __name__ == "__main__":
    raise SystemExit(main_guard(main)())
