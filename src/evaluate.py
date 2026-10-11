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
import hashlib
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import task as task_mod  # noqa: E402
from src.display import truncate  # noqa: E402
from src.config import load_policy, load_task, read_text_smart, main_guard  # noqa: E402
from src.ledger import Budget, BudgetExceeded, Ledger  # noqa: E402
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
    for lineno, raw in enumerate(read_text_smart(path).splitlines(), 1):
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


# ------------------------------------------------------------------ 执行与汇总


def run_evaluation(
    client,
    task_cfg: dict,
    rows: list[dict],
    *,
    params: dict,
    echo: bool = True,
) -> tuple[list[dict], bool]:
    """逐条跑评估集。返回 (逐条记录, 是否因预算中止)。

    【为什么把这段从 main() 里拆出来】

    拆之前，"跑评估"和"打印表格"是同一坨代码，于是它只能在真的连上模型时被验证 ——
    也就是说，你要花真金白银才能知道这段代码对不对。

    拆之后，mocktest 可以用本地假服务端把整条评估链路（含指标计算和审核表生成）
    跑一遍：不花钱、不联网、几秒钟。

    一条通用原则：**把"算"和"显示"分开，是让核心逻辑可被测试的第一步。**
    """
    records: list[dict] = []
    truncated = False

    for index, item in enumerate(rows, 1):
        item_id = str(item.get("id") or f"#{index}")
        gold = item.get("gold") or {}
        if echo:
            print(f"[{index:>3}/{len(rows)}] {item_id:<8} {truncate(item.get('title', ''), 46)}")

        record: dict = {
            "id": item_id,
            "title": item.get("title", ""),
            "gold": gold,
            # 占位样本只用来验证链路通不通，不能计入准确率。
            # 它们是合成内容（"甲实验室""乙公司"），
            # 拿真实文章去测，模型永远不可能"命中"它们 ——
            # 把它们算进分母，只会得到一个被凭空拉低的假数字。
            "placeholder": bool(item.get("_placeholder")),
        }

        try:
            messages = task_mod.build_messages(task_cfg, item)
            result = client.chat(
                messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                json_mode=bool(params.get("json_mode")),
                purpose=f"eval:{item_id}",
            )
        except BudgetExceeded as exc:
            # 预算用尽不是"这一条失败"，而是"整轮到此为止"。
            #
            # 关键在于：已经跑出来的结果必须保留。
            # 一次被中断的评估如果丢掉全部数据，你不但白花了钱，还得再花一次 ——
            # 而"怕白花钱"会让你不敢把限额调小，于是限额形同虚设。
            #
            # 这条路径是先想清楚"预算耗尽时会发生什么"才补上的。
            # 它是那种平时永远不触发、一旦触发就同时损失钱和数据的缺陷。
            if echo:
                print()
                print("  " + "-" * 64)
                print(f"  预算用尽，评估中止：{exc}")
                print(f"  已完成的 {len(records)} 条结果会照常写入，不会丢。")
                print("  " + "-" * 64)
            truncated = True
            break
        except LLMError as exc:
            record.update(status="call_failed", error=str(exc), format_ok=False, category_correct=False)
            records.append(record)
            if echo:
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

        if echo:
            flag = "OK " if format_ok else "格式"
            hit = "" if category_correct is None else (" 分类命中" if category_correct else " 分类错")
            print(f"        [{flag}] {result.latency_ms}ms{hit}")
            for problem in problems:
                print(f"          ! {problem}")

    return records, truncated


def summarize(
    records: list[dict],
    *,
    client,
    task_cfg: dict,
    set_path: Path,
    has_placeholder: bool,
    truncated: bool = False,
) -> dict:
    """把逐条记录汇成指标。纯计算，不联网，所以能被离线测试。"""
    total = len(records)

    # 准确率只在【非占位】样本上算。
    #
    # 占位样本（"甲实验室""乙公司"那些合成条目）是用来验证链路通不通的，
    # 拿真实文章去测，模型永远不可能命中它们。
    # 把它们算进分母，只会得到一个被凭空拉低的假数字 ——
    # 而一个假数字比没有数字更坏：你会照着它去优化。
    #
    # 格式合规率仍然算全部：无论输入是真是假，
    # "模型有没有按结构输出"都是一个有意义的问题。
    scored = [r for r in records if not r.get("placeholder")]
    placeholder_count = total - len(scored)

    format_ok_count = sum(1 for r in records if r.get("format_ok"))
    judged = [r for r in scored if r.get("category_correct") is not None]
    correct = sum(1 for r in judged if r["category_correct"])
    importance_errors = [r["importance_error"] for r in scored if r.get("importance_error") is not None]
    latencies = [int(r["latency_ms"]) for r in records if r.get("latency_ms")]
    costs = [r.get("cost_cny") or 0.0 for r in records]
    ok_records = [r for r in records if r.get("status") == "ok"]

    try:
        eval_set_label = str(set_path.relative_to(PROJECT_ROOT))
    except ValueError:
        eval_set_label = str(set_path)

    return {
        "total": total,
        "scored": len(scored),
        "placeholder_count": placeholder_count,
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
        "cost_known": all(r.get("cost_known") for r in ok_records) if ok_records else True,
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in records),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in records),
        "provider": client.provider_name,
        "model": client.model,
        "task_id": task_cfg.get("id"),
        "config_fingerprint": task_fingerprint(task_cfg),
        "eval_set": eval_set_label,
        "placeholder_data": has_placeholder,
        # 被预算中断的一轮，必须留下痕迹。
        # 否则一份"只跑了 6 条"的结果会在两周后看起来像"跑满了 20 条"。
        "truncated": truncated,
    }


# ------------------------------------------------------------------ 与上次对比


def task_fingerprint(task_cfg: dict) -> str:
    """任务定义的指纹 —— 用来回答"这次到底改了什么"。

    只把【会影响模型行为】的字段算进去。`_` 开头的注释字段不算：
    否则我改一句注释，你就会看到"提示词变了"，而它其实什么都没影响。
    一个乱报变化的检测，很快就会被无视。

    为什么要记这个：Rubric 里写着「一次只改一个变量」，
    但那句话以前只是一句劝告 —— 系统不会告诉你有没有违反它。
    而违反的代价是具体的：你同时改了提示词和模型，拿到一个 +15%，
    却永远不知道是哪个起的作用。**这次实验白做了。**
    """
    payload = {k: v for k, v in task_cfg.items() if not k.startswith("_")}
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def find_previous_run(*, before: str, task_id: str, eval_set: str) -> tuple[Path, dict] | None:
    """找上一次【可比】的评估结果。

    "可比"的严格定义是这段代码里最重要的部分，因为它决定了你会不会
    读出一个假的"变好了 / 变差了"：

      · 假模型（stub）跑出来的数字不能比 —— 它永远回答「其他」
      · 被预算中止的一半结果不能比 —— 分母都不一样
      · 换了任务定义、或换了评估集的，不能比 —— 尺子变了

    不符合的统统跳过。宁可说"没有可比的基线"，也不给一个错的对比。
    """
    out_dir = PROJECT_ROOT / "out" / "eval"
    if not out_dir.exists():
        return None

    candidates: list[tuple[str, Path, dict]] = []
    for path in out_dir.glob("*.json"):
        if path.stem >= before:
            continue
        try:
            data = json.loads(read_text_smart(path))
        except (json.JSONDecodeError, OSError, ValueError):
            continue
        metrics = data.get("metrics") or {}
        if metrics.get("stub") or metrics.get("truncated"):
            continue
        if metrics.get("task_id") != task_id or metrics.get("eval_set") != eval_set:
            continue
        candidates.append((path.stem, path, metrics))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, path, metrics = candidates[-1]
    return path, metrics


def render_comparison(prev: dict, cur: dict, prev_label: str) -> list[str]:
    """把"和上次比"渲染成几行。

    这里最要紧的一句话是【噪声提醒】。
    20 条样本时，1 条样本就是 5 个百分点 ——
    准确率从 55% 涨到 60%，可能只是换了一条蒙对的。
    不把这件事说出来，你会为了一个不存在的进步去改提示词，
    然后在下一次得到一个不存在的退步。
    """
    lines: list[str] = []
    judged = cur.get("category_judged") or 0
    one_sample = (100.0 / judged) if judged else None

    def delta(cur_value, prev_value, *, unit="", precision=1,
              lower_is_better=False, scale=1.0):
        """一行对比。precision 很重要 ——
        单条成本常在 0.0003 元这个量级，用一位小数会显示成「0.0 → 0.0 变差」，
        看起来像个笑话，而它其实是有意义的。"""
        if cur_value is None or prev_value is None:
            return "（上次没有这个数字）"
        c, p = cur_value * scale, prev_value * scale
        diff = c - p
        fmt = f"{{:.{precision}f}}"
        if abs(diff) < 10 ** (-precision) / 2:
            return f"{fmt.format(p)}{unit} → {fmt.format(c)}{unit}　持平"
        better = (diff < 0) if lower_is_better else (diff > 0)
        mark = "变好" if better else "变差"
        sign = "+" if diff > 0 else ""
        return (f"{fmt.format(p)}{unit} → {fmt.format(c)}{unit}　"
                f"{sign}{fmt.format(diff)}{unit}　{mark}")

    lines.append(f"  对比基线        {prev_label}")
    lines.append("    分类准确率    " + delta(
        cur.get("category_accuracy"), prev.get("category_accuracy"),
        unit="%", precision=1, scale=100.0))
    lines.append("    格式合规率    " + delta(
        cur.get("format_ok_rate"), prev.get("format_ok_rate"),
        unit="%", precision=1, scale=100.0))
    lines.append("    重要性误差    " + delta(
        cur.get("importance_mae"), prev.get("importance_mae"),
        precision=2, lower_is_better=True))
    lines.append("    单条成本      " + delta(
        cur.get("cost_avg_cny"), prev.get("cost_avg_cny"),
        unit=" 元", precision=6, lower_is_better=True))

    # ---- 这次到底动了几个变量
    #
    # Rubric 里写着「一次只改一个变量」，但那以前只是一句劝告 ——
    # 系统不会告诉你有没有违反它。而违反的代价很具体：
    # 你同时改了提示词和模型，拿到 +15%，却永远不知道是哪个起的作用。
    # **这次实验白做了**，甚至更糟：你会把一个有效的改动当成无效的丢掉。
    changed: list[str] = []
    if prev.get("config_fingerprint") != cur.get("config_fingerprint"):
        changed.append("任务定义（提示词/分类/输出结构）")
    if prev.get("provider") != cur.get("provider"):
        changed.append("厂商")
    if prev.get("model") != cur.get("model"):
        changed.append("模型")

    if not changed:
        lines.append("  本次改动了      什么都没改")
        lines.append("                  → 上面那几行差别只是随机性，")
        lines.append("                     它正好可以当「噪声有多大」的标尺。")
    elif len(changed) == 1:
        lines.append(f"  本次改动了      {changed[0]}")
        lines.append("                  → 干净的实验：只动了一个变量，结论可信。")
    else:
        lines.append("")
        lines.append(f"  !! 本次同时改了 {len(changed)} 个变量：{'、'.join(changed)}")
        lines.append("     你无法知道是哪一个起的作用 —— **这次实验白做了**。")
        lines.append("     回退掉一个，只改另一个，重跑一次。")

    # 噪声提醒：只差一条样本的，不要当结论。
    cur_acc, prev_acc = cur.get("category_accuracy"), prev.get("category_accuracy")
    if one_sample is not None and cur_acc is not None and prev_acc is not None:
        diff_samples = abs(cur_acc - prev_acc) * judged
        if diff_samples <= 1.0001:
            lines.append("")
            lines.append(f"  ! 这次只差 {diff_samples:.0f} 条样本"
                         f"（{judged} 条里的一条 = {one_sample:.0f} 个百分点）。")
            lines.append("    这个差别可能只是运气。**别据此下结论**，先把样本量做大。")
        elif judged < 50:
            lines.append("")
            lines.append(f"  ! 样本只有 {judged} 条，1 条 = {one_sample:.0f} 个百分点。")
            lines.append("    差别看起来不小，但仍然要留一点怀疑。")

    return lines


# ------------------------------------------------------------------ 主流程


def main() -> int:
    parser = argparse.ArgumentParser(description="跑评估集，得到基线数字")
    parser.add_argument("--set", "-Set", default="eval/eval_set.jsonl", help="评估集路径")
    parser.add_argument("--limit", "-Limit", "-limit", type=int, default=0, help="只跑前 N 条（0 表示全部）")
    parser.add_argument("--provider", "-Provider", default=None, help="指定厂商（默认自动选有密钥的那个）")
    parser.add_argument("--model", "-Model", default=None, help="指定模型")
    parser.add_argument("--stub", "-Stub", action="store_true", help="用假模型跑，验证链路，不需要密钥")
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

    if args.stub:
        # 【为什么 eval 也需要 --stub】
        #
        # pipeline 和 daily 都有假模型模式，eval 本来没有 —— 这是个不一致：
        # **你唯一要花钱的那一步，偏偏是唯一不能预览的那一步。**
        #
        # 一个只在花钱那一刻才第一次运行的命令，你会对它充满陌生感：
        # 不知道输出长什么样、不知道要等多久、不知道正常是什么样子。
        # 于是你迟迟不跑它 —— 而"迟迟不跑评估"正是这套系统最想避免的事。
        from src.pipeline import StubClient

        client = StubClient()
        print()
        print("  " + "!" * 64)
        print("  --stub：正在用【假模型】跑评估。")
        print("  它永远回答「其他」这个分类，所以下面的准确率【没有意义】。")
        print("  这个命令验证的是【链路】—— 指标算得对不对、审核表长什么样、")
        print("  结果文件写在哪。要真数字，去掉 --stub。")
        print("  " + "!" * 64)
    else:
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

    records, truncated = run_evaluation(client, task_cfg, rows, params=params, echo=True)

    # ------------------------------------------------------------- 汇总

    metrics = summarize(
        records,
        client=client,
        task_cfg=task_cfg,
        set_path=set_path,
        has_placeholder=has_placeholder,
        truncated=truncated,
    )
    # 假模型跑出来的数字必须能被识别出来。
    # 它和 label_origin、truncated 是同一条思路：
    # **记下来，但显式标注** —— 而不是不记，或者记了却不说是假的。
    # 不标注的后果很具体：三周后你翻 out/eval/ 找"上次基线是多少"，
    # 会翻到一份 0% 准确率的假模型结果，然后花半小时怀疑人生。
    metrics["stub"] = bool(args.stub)
    if args.stub:
        metrics["provider"] = "stub"

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
    if truncated:
        print("  本次结果（被预算中断，没有跑完）")
    elif args.stub:
        print("  本次结果（假模型，仅供预览 ——【不是】你的基线）")
    else:
        print("  本次结果（这就是你的基线）")
    print("=" * 68)
    if args.stub:
        print("  !! 上面那个数字来自假模型，它永远回答「其他」。")
        print("     链路是通的，数字是空的。真正的基线要等 .env 配好之后不带 --stub 跑。")
        print()
    if truncated:
        print(f"  !! 这一轮在第 {metrics['total']} 条之后因预算中止。")
        print("     结果文件里记着 truncated=true —— 别在两周后把这一份当成完整基线。")
        print()
    print(f"  样本数          {metrics['total']}")
    if metrics.get("placeholder_count"):
        print(f"    其中占位样本  {metrics['placeholder_count']} 条（合成内容，只用来验证链路）")
        print(f"    真正计分的    {metrics['scored']} 条  <-- 准确率只按这些算")
    print(f"  格式合规率      {metrics['format_ok']}/{metrics['total']}  =  {metrics['format_ok_rate'] * 100:.1f}%")
    if accuracy is None:
        if metrics.get("scored", 0) == 0 and metrics.get("placeholder_count"):
            print("  分类准确率      （没有可计分的样本 —— 上面那些是合成内容，算进去只会得到一个假数字）")
        else:
            print("  分类准确率      （没有真值，无法计算 —— 请先标注 gold.category）")
    else:
        print(f"  分类准确率      {metrics['category_correct']}/{metrics['category_judged']}  =  "
              f"{accuracy * 100:.1f}%   <-- 你唯一的『好坏』判据")
    if metrics["importance_mae"] is not None:
        print(f"  重要性平均误差  {metrics['importance_mae']}  （越小越好）")
    print(f"  延迟            平均 {metrics['latency_avg_ms']}ms   "
          f"中位 {metrics['latency_p50_ms']}ms   P95 {metrics['latency_p95_ms']}ms")
    if metrics["cost_known"]:
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

    # ---- 和上一次可比的结果对比
    #
    # S3 的整个循环是：改动 → 重跑 → 【比较数字】→ 留或回退。
    # 中间那一步原来是空的 —— 你得自己去 out/eval/ 翻两个 JSON 文件对照。
    # 于是"这次改动有没有用"这个问题实际上没人回答得了，
    # 而整套系统的意义恰恰在于回答它。
    if not truncated and not args.stub:
        previous = find_previous_run(
            before=stamp,
            task_id=str(task_cfg.get("id")),
            eval_set=str(metrics.get("eval_set")),
        )
        if previous is None:
            print("  对比基线        没有找到可比的基线。")
            print("                  这是第一条 —— 它的作用就是当尺子。")
            print("                  改了提示词或换了模型之后，再跑一次就会和它比。")
            print()
        else:
            prev_path, prev_metrics = previous
            for line in render_comparison(
                prev_metrics, metrics, f"out/eval/{prev_path.name}"
            ):
                print(line)
            print()

    if truncated:
        print("  下一步：调大 config/policy.json 的 budget 限额，或分批跑。")
    elif args.stub:
        print("  下一步：填 .env，跑 run.cmd probe，然后去掉 --stub 跑一次真的。")
    else:
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
        *(
            ["**这一轮被预算中断，没有跑完 —— 不要把它当成完整基线。**", ""]
            if metrics.get("truncated")
            else []
        ),
        *(
            ["**这一轮的模型是假的（--stub），内容没有意义 —— 这张表只是让你看格式。**", ""]
            if metrics.get("stub")
            else []
        ),
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
        raise SystemExit(main_guard(main)())
    except FatalLLMError as exc:
        print(f"\n[致命错误] {exc}\n")
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\n[中断] 已停止。已产生的账本记录不会丢失。\n")
        raise SystemExit(130)
