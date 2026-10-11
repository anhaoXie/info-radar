"""第 ⓪ 层 · 评估

它回答三个问题：

    1. 模型能不能稳定地产出合法结构？    自动可测  —— 格式合规率
    2. 它的判断对不对？                  需要真值  —— 分类准确率
    3. 代价是多少？                      自动可测  —— 成本、延迟

【它有意不回答什么】

"摘要有没有漏掉要点""有没有出现原文没说过的话"—— 这两件事目前只能人看。
所以本脚本会额外生成一张【人工审核表】，让你用几分钟填完，
而不是假装自动化能测这些东西。

把不可自动测的东西硬塞进自动化，你会得到一个连自己都不信的分数 ——
那比没有分数更糟，因为它会给你虚假的安心。

用法：
    python src/evaluate.py                       # 跑默认评估集
    python src/evaluate.py --limit 3             # 只跑 3 条，先验证链路通
    python src/evaluate.py --provider zhipu --model glm-5.3-flash
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import task as task_mod  # noqa: E402
from src.config import load_policy, load_task  # noqa: E402
from src.ledger import Budget, Ledger  # noqa: E402
from src.llm import FatalLLMError, LLMClient, LLMError  # noqa: E402


# ------------------------------------------------------------------ 读取


def load_eval_set(path: Path) -> tuple[list[dict], bool]:
    if not path.exists():
        raise SystemExit(
            f"找不到评估集：{path}\n"
            f"  {path.name} 是一行一条 JSON。字段格式见 eval/RUBRIC.md，"
            f"示例见仓库里自带的 eval/eval_set.jsonl。"
        )
    rows: list[dict] = []
    has_placeholder = False
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            if row.get("_placeholder"):
                has_placeholder = True
            rows.append(row)
    return rows, has_placeholder


# ------------------------------------------------------------------ 统计


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(ratio * (len(ordered) - 1))))
    return ordered[index]


# ------------------------------------------------------------------ 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description="跑评估集，得到基线数字")
    parser.add_argument("--set", "-Set", default="eval/eval_set.jsonl", help="评估集路径")
    parser.add_argument("--limit", "-Limit", "-limit", type=int, default=0, help="只跑前 N 条（0 表示全部）")
    parser.add_argument("--provider", "-Provider", default=None, help="指定厂商（默认自动选有密钥的那个）")
    parser.add_argument("--model", "-Model", default=None, help="指定模型")
    parser.add_argument("--out-dir", default="out/eval", help="结果输出目录")
    args = parser.parse_args()

    set_path = PROJECT_ROOT / args.set
    rows, has_placeholder = load_eval_set(set_path)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"评估集是空的：{set_path}")

    task_cfg = load_task()
    policy = load_policy()
    ledger = Ledger(policy)
    budget = Budget(policy, ledger=ledger)
    params = task_cfg.get("prompt", {}).get("params", {})

    try:
        client = LLMClient(
            args.provider, model=args.model, policy_cfg=policy, ledger=ledger, budget=budget
        )
    except (LLMError, RuntimeError) as exc:
        print(f"\n[错误] {exc}\n")
        return 2

    print("=" * 68)
    print(f"评估集    {set_path.relative_to(PROJECT_ROOT)}  共 {len(rows)} 条")
    print(f"任务      {task_cfg.get('id')}")
    print(f"厂商/模型 {client.provider_name} / {client.model}")
    print("=" * 68)

    if has_placeholder:
        print("\n[提醒] 这批样本里含有占位数据（_placeholder: true）。")
        print("       它能验证链路通不通，但不能代表真实效果。")
        print("       走完流程后请用你真实的 20 条资讯替换 eval/eval_set.jsonl。\n")

    records: list[dict] = []
    for index, item in enumerate(rows, 1):
        item_id = str(item.get("id") or f"#{index}")
        gold = item.get("gold") or {}
        label = str(item.get("title", ""))[:38]
        print(f"[{index:>3}/{len(rows)}] {item_id:<8} {label}")

        record: dict = {"id": item_id, "title": item.get("title", ""), "gold": gold}

        try:
            messages = task_mod.build_messages(task_cfg, item)
            result = client.chat(
                messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                json_mode=bool(params.get("json_mode")),
                purpose=f"eval:{item_id}",
            )
        except LLMError as exc:
            record.update(status="call_failed", error=str(exc), format_ok=False, category_correct=False)
            records.append(record)
            print(f"        调用失败：{str(exc)[:120]}")
            continue

        obj, parse_error = task_mod.parse_json(result.text)
        if obj is None:
            problems, warnings = [parse_error or "解析失败"], []
        else:
            problems, warnings = task_mod.validate(task_cfg, obj)

        format_ok = not problems
        category_correct = None
        if obj is not None and gold.get("category"):
            category_correct = obj.get("category") == gold["category"]

        importance_error = None
        if obj is not None and isinstance(gold.get("importance"), int):
            importance = obj.get("importance")
            if isinstance(importance, int) and not isinstance(importance, bool):
                importance_error = abs(importance - gold["importance"])

        record.update(
            status="ok" if format_ok else "format_error",
            output=obj,
            raw_text=result.text[:2000],
            problems=problems,
            warnings=warnings,
            format_ok=format_ok,
            category_correct=category_correct,
            importance_error=importance_error,
            latency_ms=result.latency_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_cny=result.cost_cny,
            cost_known=result.cost_known,
            attempts=result.attempts,
        )
        records.append(record)

        flag = "OK " if format_ok else "格式"
        hit = "" if category_correct is None else (" 分类命中" if category_correct else " 分类错")
        print(f"        [{flag}] {result.latency_ms}ms{hit}")
        for problem in problems:
            print(f"          ! {problem}")

    # ------------------------------------------------------------- 汇总

    total = len(records)
    format_ok_count = sum(1 for r in records if r.get("format_ok"))
    judged = [r for r in records if r.get("category_correct") is not None]
    correct = sum(1 for r in judged if r["category_correct"])
    importance_errors = [r["importance_error"] for r in records if r.get("importance_error") is not None]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms")]
    costs = [r.get("cost_cny") or 0.0 for r in records]
    cost_known = all(r.get("cost_known") for r in records if r.get("status") == "ok")

    metrics = {
        "total": total,
        "format_ok": format_ok_count,
        "format_ok_rate": round(format_ok_count / total, 4) if total else 0.0,
        "category_judged": len(judged),
        "category_correct": correct,
        "category_accuracy": round(correct / len(judged), 4) if judged else None,
        "importance_mae": round(statistics.mean(importance_errors), 3) if importance_errors else None,
        "latency_avg_ms": int(statistics.mean(latencies)) if latencies else 0,
        "latency_p50_ms": int(percentile(latencies, 0.50)),
        "latency_p95_ms": int(percentile(latencies, 0.95)),
        "cost_total_cny": round(sum(costs), 6),
        "cost_avg_cny": round(sum(costs) / total, 6) if total else 0.0,
        "cost_known": cost_known,
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in records),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in records),
        "provider": client.provider_name,
        "model": client.model,
        "task_id": task_cfg.get("id"),
        "eval_set": str(set_path.relative_to(PROJECT_ROOT)),
        "placeholder_data": has_placeholder,
    }

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stamp}.json"
    json_path.write_text(
        json.dumps({"metrics": metrics, "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    review_path = out_dir / f"{stamp}-review.md"
    review_path.write_text(_render_review(records, metrics, stamp), encoding="utf-8")

    ledger.record("eval_run", **metrics)

    # ------------------------------------------------------------- 打印

    accuracy = metrics["category_accuracy"]
    print()
    print("=" * 68)
    print("  本次结果（这就是你的基线）")
    print("=" * 68)
    print(f"  样本数          {total}")
    print(f"  格式合规率      {format_ok_count}/{total}  =  {metrics['format_ok_rate'] * 100:.1f}%")
    if accuracy is None:
        print("  分类准确率      （没有真值，无法计算 —— 请先标注 gold.category）")
    else:
        print(f"  分类准确率      {correct}/{len(judged)}  =  {accuracy * 100:.1f}%   <-- 你唯一的『好坏』判据")
    if metrics["importance_mae"] is not None:
        print(f"  重要性平均误差  {metrics['importance_mae']}  （越小越好）")
    print(f"  延迟            平均 {metrics['latency_avg_ms']}ms   "
          f"中位 {metrics['latency_p50_ms']}ms   P95 {metrics['latency_p95_ms']}ms")
    if cost_known:
        print(f"  成本            合计 {metrics['cost_total_cny']} 元   "
              f"单条 {metrics['cost_avg_cny']} 元")
    else:
        print(f"  Token           输入 {metrics['prompt_tokens']}  输出 {metrics['completion_tokens']}")
        print("  成本            【单价未配置，未计入】改 config/providers.json 的 price_cny_per_1m")
    print()
    print(f"  详细结果        {json_path.relative_to(PROJECT_ROOT)}")
    print(f"  人工审核表      {review_path.relative_to(PROJECT_ROOT)}   <-- 现在去看这个")
    print(f"  账本            {ledger.path.relative_to(PROJECT_ROOT)}")
    print()
    print("  下一步：把人看出来的问题，变成 eval/eval_set.jsonl 里的真值。")
    print()
    return 0


def _render_review(records: list[dict], metrics: dict, stamp: str) -> str:
    """生成人工审核表。

    设计目标只有一个：让"人看一眼"这件事便宜到你会真的去做。
    每一条只问三个能用眼睛回答的问题。
    """
    lines = [
        f"# 人工审核表 · {stamp}",
        "",
        f"- 任务：`{metrics.get('task_id')}`",
        f"- 模型：`{metrics.get('provider')} / {metrics.get('model')}`",
        f"- 样本：{metrics.get('total')} 条",
        f"- 自动判据：格式合规 {metrics.get('format_ok_rate', 0) * 100:.1f}%，"
        + (f"分类准确 {metrics.get('category_accuracy', 0) * 100:.1f}%" if metrics.get("category_accuracy") is not None else "分类准确（无真值）"),
        "",
        "> 这张表填完，你就获得了自动化测不出来的三个数字：**遗漏率、幻觉率、重要性判断的合理比例**。",
        "> 它们比准确率更重要，因为准确率错了只是分错类，幻觉错了是你会相信一件没发生的事。",
        "",
        "---",
        "",
    ]

    for index, record in enumerate(records, 1):
        output = record.get("output") or {}
        gold = record.get("gold") or {}
        model_category = output.get("category", "（无）")
        gold_category = gold.get("category", "（未标注）")
        verdict = ""
        if record.get("category_correct") is True:
            verdict = "  [分类命中]"
        elif record.get("category_correct") is False:
            verdict = "  [分类不符]"

        lines += [
            f"## {index}. `{record['id']}`{verdict}",
            "",
            f"**原文标题**　{record.get('title', '')}",
            "",
            f"**模型分类**　{model_category}　（真值：{gold_category}）　"
            f"**重要性**　{output.get('importance', '（无）')}　（真值：{gold.get('importance', '未标注')}）",
            "",
            f"**模型摘要**　{output.get('summary', record.get('raw_text', '（无输出）'))}",
            "",
            f"**它认为的影响**　{output.get('so_what', '（无）')}",
            "",
            "**它声称的原文依据**",
            "",
        ]
        evidence = output.get("evidence") or []
        if evidence:
            for piece in evidence:
                lines.append(f"> {piece}")
        else:
            lines.append("> （没有提供依据 —— 这本身就是个问题）")
        lines += [
            "",
            "**请你判断（打勾即可）**",
            "",
            "- [ ] 摘要遗漏了原文的关键点吗？漏掉的是：",
            "- [ ] 摘要里出现了原文没说过的话吗？是哪句：",
            "- [ ] 上面那些『原文依据』，真的能在原文里找到吗？找不到的是：",
            "- [ ] 重要性打分合理吗？你会给几分：",
            "",
            "---",
            "",
        ]

    lines += [
        "## 收尾：把结论变成真值",
        "",
        "审核完，打开 `eval/eval_set.jsonl`，对每一条：",
        "",
        "1. 如果分类错了 → 修正那一行的 `gold.category`（用你的判断，不是模型的判断）",
        "2. 如果原文里没有关键信息 → 说明这篇不该进评估集，删掉它",
        "3. 如果模型编了事实 → 在这条上标记 `\"hallucinated\": true`，它是你后面回归测试的重点",
        "",
        "评估集是你唯一不会骗你的东西。它值得你花那二十分钟。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FatalLLMError as exc:
        print(f"\n[致命错误] {exc}\n")
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\n[中断] 已停止。已产生的账本记录不会丢失。\n")
        raise SystemExit(130)
