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
    print(stage_line("S1", "评估层 · 基线与判据", "OK" if results else "待办"))
    if results:
        latest = results[-1]
        try:
            payload = json.loads(latest.read_text(encoding="utf-8"))
            metrics = payload.get("metrics", {})
            accuracy = metrics.get("category_accuracy")
            accuracy_text = f"{accuracy * 100:.1f}%" if accuracy is not None else "无真值"
            placeholder = "  [占位数据，请替换]" if metrics.get("placeholder_data") else ""
            print(f"       最近一次 {latest.name}{placeholder}")
            print(
                f"       格式合规 {metrics.get('format_ok_rate', 0) * 100:.1f}%   "
                f"分类准确 {accuracy_text}   单条成本 {metrics.get('cost_avg_cny', 0)} 元"
            )
            print(
                f"       模型 {metrics.get('provider')} / {metrics.get('model')}   "
                f"延迟 P50 {metrics.get('latency_p50_ms')}ms"
            )
        except (json.JSONDecodeError, OSError):
            print("       （最近一次结果读不出来，可能写坏了）")
    else:
        print("       还没有基线。跑 run.cmd eval -Limit 3 先验证链路")

    # ---- S2 / S3 / S4
    raw_count = count_files("data/raw")
    print(stage_line("S2", "数据层 · 抓取与去重", "OK" if raw_count else "待办"))
    print(f"       data/raw 里 {raw_count} 个文件")

    digest_count = count_files("out/digests")
    print(stage_line("S3", "工作流层 · 编排与迭代", "OK" if digest_count else "待办"))
    print(stage_line("S4", "分发层 · 审核与归档", "进行中" if digest_count else "待办"))
    print(stage_line("S5", "权限层 · 限额与无人值守", "待办"))

    print()
    print(hr())
    print("  下一步看 README.md 的『三步跑起来』")
    print()
    return 0


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
    print("  run.cmd doctor          环境自检（不联网，出问题先跑这个）")
    print("  run.cmd selftest        离线自检全部逻辑（不联网，不花钱）")
    print("  run.cmd probe           探测模型接口（联网，验证真实可用性）")
    print("  run.cmd eval            跑评估集，得到基线数字")
    print("  run.cmd eval -Limit 3   只跑 3 条，先验证链路通不通")
    print("  run.cmd sources         检查信息源是否还活着（S2 阶段）")
    print("  run.cmd ledger          看今天的账本（花了多少、慢不慢）")
    print("  run.cmd status          看当前进展全貌")
    print()
    print("  当前状态：S0（地基）—— 等你在 .env 里填 key")
    print()
    return 0


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
