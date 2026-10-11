"""进展总览与账本速览。

【为什么这部分逻辑在 Python 里，而不是在 run.cmd 里】

因为它要读 JSON、算统计、格式化数字，还要处理"文件不存在"这类分支。
批处理脚本做这些事会变成一堆引号和转义，而且无法测试。

一条通用的判断：**逻辑写在你最擅长调试的语言里，胶水写在你必须用的语言里。**
run.cmd 只负责一件事 —— 把命令转交给 Python。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

WIDTH = 64


def hr(char: str = "=") -> str:
    return char * WIDTH


def find_eval_results() -> list[Path]:
    out_dir = PROJECT_ROOT / "out" / "eval"
    if not out_dir.is_dir():
        return []
    return sorted(
        [p for p in out_dir.glob("*.json") if not p.name.endswith("-review.json")],
        key=lambda p: p.stat().st_mtime,
    )


def count_files(relative: str) -> int:
    directory = PROJECT_ROOT / relative
    if not directory.is_dir():
        return 0
    return len([p for p in directory.iterdir() if p.is_file() and p.name != ".gitkeep"])


def count_inbox_files() -> int:
    """收件箱里真正会被导入的文件数。

    必须排除 _ 开头的文件，和导入器用同一套规则 ——
    否则状态面板会说"待导入 1 篇"，而那个文件永远不会被导入。
    一个会撒谎的状态面板，比没有状态面板更坏。
    """
    directory = PROJECT_ROOT / "data" / "inbox"
    if not directory.is_dir():
        return 0
    return len(
        [
            p
            for p in directory.iterdir()
            if p.is_file()
            and not p.name.startswith("_")
            and p.suffix.lower() in {".md", ".markdown", ".txt", ".html", ".htm", ".json"}
        ]
    )


def stage_line(stage: str, name: str, status: str) -> str:
    return f"  {stage:<4} {name:<28} [{status}]"


def show_progress() -> int:
    print()
    print(f"  info-radar 进展")
    print(hr())
    print()

    # ---- S0
    env_path = PROJECT_ROOT / ".env"
    key_count = 0
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            if value.strip() and ("KEY" in name.upper() or "TOKEN" in name.upper()):
                key_count += 1
    print(stage_line("S0", "地基 · 密钥与账本", "OK" if key_count else "待办"))
    print(f"       .env {'存在' if env_path.exists() else '不存在'}，已填 key {key_count} 个")
    print(f"       账本文件 {count_files('out/logs')} 个")

    # ---- S1
    results = find_eval_results()
    state = eval_set_state()
    rows = count_eval_set_rows()

    # 先把最近一次结果的 metrics 读出来，再决定 S1 该显示什么。
    # 顺序反了会得到一个具体的坏结果：跑一次 eval --stub 预览，
    # S1 就变成 [OK]，看起来像你已经有了基线。
    latest: Path | None = results[-1] if results else None
    metrics: dict = {}
    if latest is not None:
        try:
            metrics = json.loads(latest.read_text(encoding="utf-8")).get("metrics", {})
        except (json.JSONDecodeError, OSError):
            metrics = {}

    stub_only = bool(metrics.get("stub"))
    if results and not stub_only:
        s1_status = "有基线"
    elif stub_only:
        s1_status = "仅预览"
    elif state == "real":
        s1_status = "有评估集"
    else:
        s1_status = "待办"
    print(stage_line("S1", "评估层 · 基线与判据", s1_status))

    label_state = "real" if state == "real" else ("占位" if state == "placeholder" else "缺失")
    print(f"       评估集 {row_count_text(rows)}（{label_state}）")
    if state == "placeholder" and not (PROJECT_ROOT / "eval" / "candidates.jsonl").exists():
        # 只有在"既没有真值、也没有待标注样本"时，才提示去搞文档。
        # 已经有 20 条样本摆在那儿了还让人去导入，是在制造多余的步骤。
        print("       ^ 现在是占位数据，只能验证链路，不能代表真实效果")
        print("         先用 run.cmd collect 抓一批，或 run.cmd inbox 用你自己的文档")

    labels_now = label_progress()
    if labels_now and labels_now[0] < labels_now[1]:
        filled, total = labels_now
        print(f"       标注进度 {filled}/{total}   <- 填完跑 run.cmd compile")
    elif labels_now:
        print(f"       标注进度 {labels_now[0]}/{labels_now[1]}   <- 已填完，跑 run.cmd compile")

    if latest is not None:
        accuracy = metrics.get("category_accuracy")
        accuracy_text = f"{accuracy * 100:.1f}%" if accuracy is not None else "无真值"
        placeholder = "  [占位数据]" if metrics.get("placeholder_data") else ""
        print(f"       最近一次 {latest.name}{placeholder}")
        if metrics.get("stub"):
            print("       !! 这一次用的是【假模型】，数字没有意义 —— 这不是你的基线")
        if metrics.get("truncated"):
            print("       !! 这一轮被预算中断，只跑了一部分 —— 不是完整基线")
        print(
            f"       格式合规 {metrics.get('format_ok_rate', 0) * 100:.1f}%   "
            f"分类准确 {accuracy_text}   单条成本 {metrics.get('cost_avg_cny', 0)} 元"
        )
        print(
            f"       模型 {metrics.get('provider')} / {metrics.get('model')}   "
            f"延迟 P50 {metrics.get('latency_p50_ms')}ms"
        )

    # ---- S2
    raw_count = count_files("data/raw")
    inbox_count = count_inbox_files()
    print(stage_line("S2", "数据层 · 抓取与去重", "OK" if raw_count else "待办"))
    print(f"       库内文档 {raw_count} 篇   data/inbox 待导入 {inbox_count} 篇")

    source_state = load_sources_state()
    if source_state:
        broken = {
            name: info
            for name, info in source_state.items()
            if not info.get("ok")
        }
        healthy = len(source_state) - len(broken)
        print(f"       信息源 {healthy}/{len(source_state)} 正常")
        for name, info in broken.items():
            streak = int(info.get("consecutive_failures") or 0)
            print(f"         [!!] {name}  连续失败 {streak} 次")
            if info.get("error"):
                print(f"              {str(info['error']).splitlines()[0][:80]}")
        if broken:
            print("       ^ 静态失败不会自己恢复。跑 run.cmd sources 复查，或把它停用。")

    # ---- S3 / S4 / S5
    summary = index_summary()
    if summary and summary.get("processed"):
        s3_status = "OK" if summary.get("failed", 0) == 0 else "有失败"
    elif count_results():
        s3_status = "OK"
    else:
        s3_status = "待办"
    print(stage_line("S3", "工作流层 · 编排与迭代", s3_status))
    if summary:
        print(f"       库内 {summary['total']} 篇：已处理 {summary['processed']}"
              f"（成功 {summary['ok']}，失败 {summary['failed']}），待处理 {summary['pending']}")
        if summary.get("failed"):
            print("       ^ 失败的条目已被标记为已处理，不会自己重试。")
            print("         确认要重试就跑：run.cmd pipeline --retry-failed")
    else:
        print("       还没有处理过任何条目。跑 run.cmd pipeline 或先 --stub 看看流程")
    print(f"       历史产出 {count_results()} 份 out/results/")

    digest_count = count_files("out/digests")
    final_count = len(list((PROJECT_ROOT / "out" / "digests").glob("*-final.md"))) if digest_count else 0
    review_count = count_files("out/review")
    if final_count:
        s4_status = "OK"
    elif digest_count:
        s4_status = "待审核"
    else:
        s4_status = "待办"
    print(stage_line("S4", "分发层 · 审核与归档", s4_status))
    if digest_count:
        print(f"       日报 {digest_count} 份（终版 {final_count} 份），审核表 {review_count} 份")
        if digest_count > final_count:
            newest = sorted((PROJECT_ROOT / "out" / "digests").glob("*.md"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
            pending = next((p for p in newest if not p.stem.endswith("-final")), None)
            if pending:
                print(f"       待审核：out/review/{pending.stem}.txt")
                print(f"       填完跑：run.cmd review --apply {pending.stem}")
    else:
        print("       还没有日报。跑 run.cmd pipeline 之后再 run.cmd digest")

    eval_origin = eval_set_origins()
    if eval_origin:
        parts = "，".join(f"{k} {v}" for k, v in eval_origin.items())
        print(f"       评估集来源：{parts}")

    health = health_summary()
    alert_exists = (PROJECT_ROOT / "out" / "ALERT.md").exists()
    s5_status = "有警报" if alert_exists else ("OK" if health else "待办")
    print(stage_line("S5", "权限层 · 限额与无人值守", s5_status))
    if health:
        print(f"       运行记录 {health['run_count']} 次")
        if health["last_run"]:
            last = health["last_run"]
            skipped = f"　跳过 {last.get('skipped')} 步" if last.get("skipped") else ""
            print(f"       上次运行 {last.get('ts', '?')}　"
                  f"{'成功' if last.get('ok') else '失败'}　用时 {last.get('seconds')}s{skipped}")
        if health["escalating"]:
            print(f"       连续失败达阈值的任务：{'、'.join(health['escalating'])}")
    else:
        print("       还没有运行记录。跑一次：run.cmd daily --stub")

    if alert_exists:
        print()
        print("  " + "!" * 62)
        print("  !! 存在未处理的警报：out/ALERT.md")
        print("     有任务连续失败达到阈值。这不是「今天没新闻」，是「流程坏了」。")
        print("     跑 run.cmd health 看是哪个任务、连续失败了几次。")
        print("  " + "!" * 62)

    print()
    print(hr())
    # 下一步由数据决定，不写死。
    if labels_now and labels_now[0] < labels_now[1]:
        filled, total = labels_now
        print(f"  下一步：打开 eval/worksheet-lite.md 扫一遍（标题+开头，5 分钟）")
        print(f"         （当前 {filled}/{total}）填完跑 run.cmd compile")
    elif state == "placeholder" and not health:
        print("  下一步：run.cmd collect 抓一批文档，或 run.cmd inbox 用你自己的文档")
    elif state != "real":
        print("  下一步：run.cmd sample -N 20")
    elif not results:
        print("  下一步：run.cmd eval -Limit 3  先花几分钱验证链路")
    else:
        print("  下一步：run.cmd eval  跑完整基线")
    print()
    return 0


def label_progress() -> tuple[int, int] | None:
    """标注进度。返回 (已填, 总数)，没有抽样任务时返回 None。

    "下一步该做什么"这件事必须由数据决定，不能写死。
    第一版把下一步写成了固定的 "run.cmd inbox"，
    于是即使 20 条样本已经抽好摆在那里，它还在让你先去导入文档。
    """
    candidates_path = PROJECT_ROOT / "eval" / "candidates.jsonl"
    labels_path = PROJECT_ROOT / "eval" / "labels.txt"
    if not candidates_path.exists():
        return None
    try:
        total = len([ln for ln in candidates_path.read_text(encoding="utf-8").splitlines() if ln.strip()])
    except OSError:
        return None
    filled = 0
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "|" not in stripped:
                continue
            parts = [p.strip() for p in stripped.split("|")]
            if len(parts) >= 3 and parts[1] and parts[2]:
                filled += 1
    return filled, total


def row_count_text(count: int) -> str:
    return f"{count} 条" if count else "空"


def show_ledger() -> int:
    today = datetime.now().strftime("%Y-%m-%d")
    path = PROJECT_ROOT / "out" / "logs" / f"ledger-{today}.jsonl"
    print()
    if not path.exists():
        print(f"  今天还没有账本记录（{path.name}）。")
        print("  跑一次 run.cmd eval -Limit 3 就会有了。")
        print()
        return 0

    calls: list[dict] = []
    evals: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "llm_call":
            calls.append(row)
        elif row.get("event") == "eval_run":
            evals.append(row)

    total_cost = sum(float(r.get("cost_cny") or 0) for r in calls)
    prompt_tokens = sum(int(r.get("prompt_tokens") or 0) for r in calls)
    completion_tokens = sum(int(r.get("completion_tokens") or 0) for r in calls)
    latencies = [int(r.get("latency_ms") or 0) for r in calls]
    retried = [r for r in calls if int(r.get("attempts") or 1) > 1]
    cost_known = all(r.get("cost_known") for r in calls) if calls else True

    print(f"  账本  {path.name}")
    print("  " + hr("-"))
    print(f"  模型调用     {len(calls)} 次")
    if cost_known:
        print(f"  累计成本     {total_cost:.6f} 元")
    else:
        print(f"  累计成本     （部分单价未配置，未计入）")
    print(f"  累计 token   输入 {prompt_tokens} / 输出 {completion_tokens}")
    if latencies:
        ordered = sorted(latencies)
        print(
            f"  延迟         平均 {sum(latencies) // len(latencies)}ms   "
            f"中位 {ordered[len(ordered) // 2]}ms   最大 {ordered[-1]}ms"
        )
    if retried:
        print(f"  发生过重试   {len(retried)} 次  <-- 值得看一眼，可能是网络或限流")

    # 按用途分组：这是"哪一步在花钱"的唯一答案
    by_purpose: dict[str, dict] = {}
    for row in calls:
        purpose = str(row.get("purpose") or "(未标注)")
        bucket = by_purpose.setdefault(purpose, {"n": 0, "cost": 0.0, "tokens": 0})
        bucket["n"] += 1
        bucket["cost"] += float(row.get("cost_cny") or 0)
        bucket["tokens"] += int(row.get("total_tokens") or 0)
    if len(by_purpose) > 1:
        print()
        print("  按用途分解（优化成本时先看这里）")
        for purpose, bucket in sorted(by_purpose.items(), key=lambda kv: -kv[1]["cost"]):
            print(f"    {purpose:<20} {bucket['n']:>4} 次   {bucket['cost']:.6f} 元   {bucket['tokens']} token")

    print(f"  评估次数     {len(evals)}")
    print("  " + hr("-"))
    print()
    return 0


def show_help() -> int:
    print()
    print("  info-radar · 个人 AI 工作系统")
    print("  " + hr("-"))
    print("  先跑这三个（不联网，不花钱，出错先看它们）")
    print("    run.cmd status          我走到哪了")
    print("    run.cmd selftest        离线自检纯逻辑（65 项，毫秒级）")
    print("    run.cmd mocktest        全链路自检：本地假服务端验证真客户端（68 项）")
    print("    run.cmd doctor          环境自检")
    print()
    print("  模型（要联网，要密钥）")
    print("    run.cmd probe           探测模型接口，列出你账号真实可用的模型")
    print("    run.cmd eval -Limit 3   花几分钱跑 3 条，验证链路")
    print("    run.cmd eval            跑完整评估，得到基线数字")
    print()
    print("  数据（S1 / S2）")
    print("    run.cmd inbox           导入 data/inbox/ 里的本地文档")
    print("    run.cmd collect         抓取 config/sources.json 里的 RSS 源")
    print("    run.cmd sources         检查信息源是否还活着")
    print()
    print("  工作流（S3）")
    print("    run.cmd pipeline --dry-run    只看要处理哪些，不发请求、不要密钥")
    print("    run.cmd pipeline --stub       用假模型跑通链路，不花钱")
    print("    run.cmd pipeline              真正处理（要密钥）")
    print("    run.cmd pipeline --retry-failed  重试之前失败的条目")
    print()
    print("  分发与审核（S4）")
    print("    run.cmd digest                从最新一轮结果生成日报 + 审核表")
    print("    run.cmd review                看历史产出")
    print("    run.cmd review --apply <时间戳>  应用审核：生成终版 + 回流评估集")
    print("    run.cmd review --stats        评估集来源统计")
    print()
    print("  自动化（S5）")
    print("    run.cmd health                看任务健康表（不联网、不要密钥，每天先看它）")
    print("    run.cmd daily --stub          用假模型跑一遍完整流程")
    print("    run.cmd daily --no-collect    只处理 inbox，不抓 RSS")
    print("    run.cmd daily                 真正跑一次（要密钥）")
    print("    run.cmd daily --continue-on-error  失败后继续（默认会停下）")
    print()
    print("  标注（S1 的核心工作，唯一不能交给 AI 的一步）")
    print("    run.cmd sample -N 20    抽 20 条，生成标注表和阅读工作纸")
    print("    run.cmd labels          看标注进度")
    print("    run.cmd compile         把标注编译成 eval/eval_set.jsonl")
    print()
    print("  账目")
    print("    run.cmd ledger          今天花了多少、慢不慢、哪一步在花钱")
    print()
    print("  当前状态：S0（地基）—— 等你在 .env 里填 key")
    print("  必读：docs/03-完整走一遍.md  ← 手把手走查，从这里开始")
    print("        docs/00-架构蓝图.md    ← 六层结构的心智模型")
    print()
    return 0


def load_sources_state() -> dict:
    path = PROJECT_ROOT / "data" / "cache" / "sources-state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("sources", {})
    except (json.JSONDecodeError, OSError):
        return {}


def eval_set_state() -> str:
    """判断评估集是占位数据、真实数据，还是根本没有。"""
    path = PROJECT_ROOT / "eval" / "eval_set.jsonl"
    if not path.exists():
        return "missing"
    text = path.read_text(encoding="utf-8")
    return "placeholder" if "_placeholder" in text else "real"


def count_eval_set_rows() -> int:
    path = PROJECT_ROOT / "eval" / "eval_set.jsonl"
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")])


def index_summary() -> dict:
    """从索引里读出"处理到哪了"。

    这一层信息不放在日志里，而是放在数据里 ——
    日志会被滚掉，而"这条处理过没有"是必须长期记得的事实。
    """
    path = PROJECT_ROOT / "data" / "cache" / "index.json"
    if not path.exists():
        return {}
    try:
        items = json.loads(path.read_text(encoding="utf-8")).get("items", {})
    except (json.JSONDecodeError, OSError):
        return {}
    processed = [m for m in items.values() if m.get("processed")]
    return {
        "total": len(items),
        "processed": len(processed),
        "ok": sum(1 for m in processed if m.get("status") == "ok"),
        "failed": sum(1 for m in processed if m.get("status") != "ok"),
        "pending": len(items) - len(processed),
    }


def count_results() -> int:
    directory = PROJECT_ROOT / "out" / "results"
    if not directory.is_dir():
        return 0
    return len(list(directory.glob("*.jsonl")))


def health_summary() -> dict:
    """读健康表。这一层信息决定面板能不能回答"它现在还活着吗"。"""
    path = PROJECT_ROOT / "data" / "cache" / "health.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    tasks = data.get("tasks", {})
    escalating = [
        name
        for name, record in tasks.items()
        if int(record.get("consecutive_failures") or 0) >= escalation_threshold()
    ]
    runs = data.get("runs", [])
    return {
        "tasks": tasks,
        "escalating": escalating,
        "last_run": runs[-1] if runs else None,
        "run_count": len(runs),
        "alert_file": (PROJECT_ROOT / "out" / "ALERT.md").exists(),
    }


def escalation_threshold() -> int:
    try:
        policy = json.loads((PROJECT_ROOT / "config" / "policy.json").read_text(encoding="utf-8"))
        return int(policy.get("health", {}).get("escalate_after_consecutive_failures", 3))
    except (json.JSONDecodeError, OSError, ValueError):
        return 3


def eval_set_origins() -> dict:
    """评估集里每条样本是怎么来的。

    分开统计"手工标注"和"审核回流"是有意义的：
    如果审核回流的样本上准确率明显更高，那说明的不是模型好，
    是你审得不够认真 —— 因为那些"真值"其实是模型自己的答案。
    """
    path = PROJECT_ROOT / "eval" / "eval_set.jsonl"
    if not path.exists():
        return {}
    counts: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = "占位数据" if row.get("_placeholder") else str(row.get("label_origin") or "手工标注")
        counts[name] = counts.get(name, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--help", action="store_true")
    parser.add_argument("--ledger", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if args.help:
        return show_help()
    if args.ledger:
        return show_ledger()
    return show_progress()


if __name__ == "__main__":
    raise SystemExit(main())
