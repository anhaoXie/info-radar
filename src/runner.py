"""第 ⑤ 层 · 权限与自动化：从"能用"到"敢交给它"

【这一层要解决的问题不是"怎么定时"，是另一个问题】

定时很简单，操作系统的计划任务就够了。
难的是：**当它无人值守地跑了几周之后，你怎么知道它还在正常工作？**

因为这两种情况在表面上完全一样：

    「今天没新闻。」          —— 系统正常，只是没有新内容
    「流程三周前就死了。」    —— 系统早就坏了，你一直没发现

两者都表现为：没有新产出。前者无害，后者意味着你可能已经在用错的产出做决定。

所以这一层的全部内容是**留下"它还活着吗"的证据**：

    1. 每次运行都记一笔：成没成、结果是什么、连续失败几次
    2. 连续失败到阈值就**大声升级**（写一个显眼的文件），不是安静地跳过
    3. 恢复了要能自己撤销警报 —— 否则警报会变成背景噪音，然后被无视

【一个必须说清的设计：默认遇到失败就停】

`run_daily` 默认在第一个失败的步骤停下，想继续要显式加 `--continue-on-error`。

为什么默认是停：因为**一个在失败之后继续产出的流水线，它的产出和有成功时长得一模一样。**
采集挂了、继续用旧数据生成日报 —— 那份日报看起来完全正常，日期是今天，
但它反映的是三天前的事实。这种错误不会有人发现，直到有人根据它做了决定。

这和我在这套系统里一直用的顺序是同一条原则：
    默认拒绝出网、默认不重试、默认不继续 —— **安全的那个选项做默认，冒险的那个要显式选。**

【为什么"只读检查"要排在"需要凭证的操作"前面】

这条在 S3 就立下了（--dry-run 不需要密钥）。这里延续：
`run.cmd health` 只看一张表，不联网、不需要密钥、不花钱。
它应该是你每天第一个跑的命令。

用法：
    python src/runner.py --daily                 # 跑一次完整流程
    python src/runner.py --daily --stub          # 用假模型跑，不需要密钥
    python src/runner.py --daily --no-collect    # 跳过 RSS 抓取（只用 inbox）
    python src/runner.py --health                # 看健康表（不联网、不要密钥）
    python src/runner.py --reset <任务名>         # 清掉某个任务的失败计数
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_policy, load_task  # noqa: E402
from src.secrets import PROJECT_ROOT as ROOT  # noqa: E402

HEALTH_PATH = ROOT / "data" / "cache" / "health.json"
ALERT_PATH = ROOT / "out" / "ALERT.md"

DEFAULT_ESCALATE_AFTER = 3
DEFAULT_KEEP_RUNS = 30


# ------------------------------------------------------------------ 健康记录


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _load_json(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Health:
    """一张"它还活着吗"的表。

    数据里记，不是日志里记 —— 日志会被滚掉，而且日志回答不了
    "这个任务连续失败了几次"这种聚合问题。
    """

    def __init__(
        self,
        path: Path | None = None,
        *,
        escalate_after: int = DEFAULT_ESCALATE_AFTER,
        keep_runs: int = DEFAULT_KEEP_RUNS,
    ) -> None:
        self.path = path or HEALTH_PATH
        self.escalate_after = max(1, int(escalate_after))
        self.keep_runs = max(1, int(keep_runs))
        self.data = _load_json(self.path, {"tasks": {}, "runs": []})
        self.data.setdefault("tasks", {})
        self.data.setdefault("runs", [])

    # ---------------------------------------------------------- 读

    @property
    def tasks(self) -> dict:
        return self.data["tasks"]

    def get(self, name: str) -> dict:
        return self.tasks.get(name, {})

    def table(self) -> list[dict]:
        rows = []
        for name, record in sorted(self.tasks.items()):
            rows.append(
                {
                    "task": name,
                    "consecutive_failures": int(record.get("consecutive_failures") or 0),
                    "last_success": record.get("last_success"),
                    "last_attempt": record.get("last_attempt"),
                    "last_summary": record.get("last_summary"),
                    "last_error": record.get("last_error"),
                }
            )
        return rows

    def escalating(self) -> list[str]:
        """连续失败达到阈值的任务名。"""
        return [
            name
            for name, record in self.tasks.items()
            if int(record.get("consecutive_failures") or 0) >= self.escalate_after
        ]

    # ---------------------------------------------------------- 写

    def success(self, name: str, summary: str = "") -> None:
        record = self.tasks.setdefault(name, {})
        record["last_attempt"] = _now()
        record["last_success"] = record["last_attempt"]
        # 成功就清零。恢复必须能自动撤销警报，否则警报会变成背景噪音。
        record["consecutive_failures"] = 0
        record["last_summary"] = summary
        record["last_error"] = None
        self.save()

    def failure(self, name: str, error: str) -> None:
        record = self.tasks.setdefault(name, {})
        record["last_attempt"] = _now()
        record["consecutive_failures"] = int(record.get("consecutive_failures") or 0) + 1
        record["last_error"] = str(error)[:500]
        self.save()

    def skip(self, name: str, reason: str) -> None:
        """记一次"没事可做"。

        计数器归零 —— 因为它回答的问题是「这个任务现在是不是坏的？」，
        而一个跑完了逻辑、判断出"没有新内容"的任务不是坏的。

        如果这里不归零，你就得到一个长期悬挂的警报；
        如果这里计一次失败，连续三天没新闻就会触发假警报。
        两种做法都会导致同一个坏结果：**你开始无视告警。**
        """
        record = self.tasks.setdefault(name, {})
        record["last_attempt"] = _now()
        record["consecutive_failures"] = 0
        record["last_summary"] = f"跳过：{reason}"
        self.save()

    def append_run(self, entry: dict) -> None:
        runs = self.data["runs"]
        runs.append(entry)
        # 只留最近 N 次。健康表是"最近怎么样"，不是永久档案 ——
        # 永久档案会让这个文件慢慢变成没人看得完的东西。
        self.data["runs"] = runs[-self.keep_runs :]
        self.save()

    def save(self) -> None:
        self.data["updated_at"] = _now()
        _atomic_write_json(self.path, self.data)

    def reset(self, name: str) -> bool:
        if name not in self.tasks:
            return False
        self.tasks[name]["consecutive_failures"] = 0
        self.tasks[name]["last_error"] = None
        self.save()
        return True

    # ---------------------------------------------------------- 警报

    def write_alert(self, names: list[str]) -> Path | None:
        if not names:
            return None
        lines = [
            "# 需要处理：有任务连续失败",
            "",
            f"生成时间　{_now()}",
            "",
            "| 任务 | 连续失败 | 上次成功 | 最近错误 |",
            "|---|---|---|---|",
        ]
        for name in names:
            record = self.get(name)
            last_success = record.get("last_success") or "（从未成功过）"
            error = str(record.get("last_error") or "").splitlines()[0][:80]
            lines.append(
                f"| `{name}` | {record.get('consecutive_failures')} 次 | {last_success} | {error} |"
            )
        lines += [
            "",
            "---",
            "",
            "「今天没新闻」和「流程三周前就死了」在表面上完全一样 —— 都是没有新产出。",
            "这张表是你区分它们的唯一依据。",
            "",
            "## 接下来怎么办",
            "",
            "1. 看上面的「最近错误」，它一般会直接说明原因（密钥失效？域名被拦？源改版了？）",
            "2. 修好之后跑一次 `run.cmd daily`，连续失败次数会自动清零，这个文件会被删除",
            "3. 如果这个任务你暂时不想管，把它停用掉，而不是留着它天天报警",
            "",
            "**不要习惯一个长期存在的警报文件。** 那等于没有警报。",
            "",
        ]
        ALERT_PATH.parent.mkdir(parents=True, exist_ok=True)
        ALERT_PATH.write_text("\n".join(lines), encoding="utf-8")
        return ALERT_PATH

    def clear_alert(self) -> bool:
        if ALERT_PATH.exists():
            ALERT_PATH.unlink()
            return True
        return False


# ------------------------------------------------------------------ 编排


class StepSkipped(Exception):
    """这一步没有事情可做 —— 但这不是失败。

    【为什么必须把它和失败分开】

    这一层存在的全部理由是区分「今天没新闻」和「流程三周前就死了」。
    那么这两件事在【代码里】也必须是两个东西。

    如果"没有新内容"被记成一次失败，连续三天没有新闻就会触发一次假警报。
    假警报的代价不是打扰你一次 —— 是你会开始无视告警，
    于是真正的那次也被无视了。

    一个只会喊"狼来了"却没有狼的警报系统，比没有警报更坏。
    """


def run_daily(
    steps: list[tuple[str, Callable[[], str]]],
    *,
    health: Health,
    continue_on_error: bool = False,
    echo: bool = True,
) -> dict:
    """按顺序跑步骤，并记录每一步的健康状况。

    steps 是 [(名字, 无参函数)]，函数返回一句人能读的摘要；
    失败就抛异常，没事可做就抛 StepSkipped。

    这个签名是刻意设计成"只依赖一个函数列表"的：
    于是测试可以传几个假的步骤进去，把"某一步失败之后会发生什么"
    完整地演一遍 —— 不需要网络、不需要密钥、不需要等真的出事。
    """
    outcome = {"ok": True, "degraded": False, "stopped_at": None, "steps": []}
    started = time.perf_counter()

    for name, fn in steps:
        step_started = time.perf_counter()
        if echo:
            print(f"\n  > {name}")

        try:
            summary = fn() or "完成"
        except StepSkipped as exc:
            elapsed = time.perf_counter() - step_started
            health.skip(name, str(exc))
            outcome["steps"].append(
                {"name": name, "ok": True, "skipped": True,
                 "summary": f"跳过：{exc}", "seconds": round(elapsed, 1)}
            )
            if echo:
                print(f"    - 跳过：{exc}   （{elapsed:.1f}s）")
            continue
        except Exception as exc:  # noqa: BLE001 - 编排层要接住任何步骤的失败
            error_text = f"{type(exc).__name__}: {exc}"
            health.failure(name, error_text)
            elapsed = time.perf_counter() - step_started
            outcome["ok"] = False
            outcome["steps"].append(
                {"name": name, "ok": False, "error": str(exc), "seconds": round(elapsed, 1)}
            )
            if echo:
                print()
                print("  " + "!" * 64)
                print(f"  x {name} 失败")
                print()
                for line in str(exc).splitlines():
                    print(f"    {line}")
                print("  " + "!" * 64)

            # 这里第一版写错了：break 放在了 if not continue_on_error 的外面，
            # 于是那个参数只决定要不要多打印一句话，行为上完全没生效。
            # 一个"看起来能配、其实不起作用"的开关比没有这个开关更坏 ——
            # 你会以为自己已经处理了这种情况。
            # 抓到它的不是读代码，是那条断言 calls == ["a","b","c"] 的测试。
            if not continue_on_error:
                outcome["stopped_at"] = name
                if echo:
                    print()
                    print("  已停止，不再执行后面的步骤。")
                    print("  理由：一个在失败之后继续产出的流水线，它的产出和有成功时【长得一模一样】。")
                    print("  用旧数据生成的日报日期是今天，反映的却是三天前的事实 —— 而且没人会发现。")
                    print()
                    print("  确实想继续，就加 --continue-on-error。")
                break

            if echo:
                print("    （--continue-on-error：继续执行后面的步骤）")
            continue

        elapsed = time.perf_counter() - step_started
        health.success(name, str(summary))
        outcome["steps"].append(
            {"name": name, "ok": True, "summary": str(summary), "seconds": round(elapsed, 1)}
        )
        if echo:
            print(f"    OK {summary}   （{elapsed:.1f}s）")

    outcome["seconds"] = round(time.perf_counter() - started, 1)
    outcome["ts"] = _now()
    outcome["skipped"] = sum(1 for s in outcome["steps"] if s.get("skipped"))
    health.append_run(outcome)

    # 警报的升级与撤销，都放在最后统一处理 ——
    # 这样无论中途停在哪一步，警报状态都反映的是"当前真实的连续失败次数"。
    escalating = health.escalating()
    if escalating:
        path = health.write_alert(escalating)
        outcome["alert"] = str(path) if path else None
    else:
        if health.clear_alert():
            outcome["alert_cleared"] = True

    return outcome


# ------------------------------------------------------------------ 真实步骤


def _step_collect(limit: int = 0) -> str:
    from src.collect import collect

    outcome = collect(limit=limit, verbose=False)
    if outcome.get("fatal"):
        raise RuntimeError(outcome["fatal"])
    if outcome["total_count"] and outcome["failed_count"] == outcome["total_count"]:
        # 全部源都失败 —— 这是"大声失败"最该响的一次。
        # 它不该被当成"今天没有新内容"。
        details = "\n".join(
            f"{r['source']}：{str(r['error']).splitlines()[0]}" for r in outcome["results"] if not r["ok"]
        )
        raise RuntimeError(f"所有信息源都失败了（{outcome['failed_count']} 个）：\n{details}")
    stored = outcome["stats"]["stored"]
    failed = outcome["failed_count"]
    suffix = f"，其中 {failed} 个源失败" if failed else ""
    return f"新增 {stored} 条，成功源 {outcome['ok_count']}/{outcome['total_count']}{suffix}"


def _step_pipeline(*, stub: bool, limit: int = 0, provider: str | None = None,
                   model: str | None = None) -> tuple[str, Path | None]:
    from src.ledger import Budget, Ledger
    from src.llm import LLMClient
    from src.pipeline import Pipeline, StubClient

    task_cfg = load_task()
    policy = load_policy()

    if stub:
        client = StubClient()
    else:
        ledger = Ledger(policy)
        budget = Budget(policy, ledger=ledger)
        client = LLMClient(provider, model=model, policy_cfg=policy, ledger=ledger, budget=budget)

    pipeline = Pipeline(client, task_cfg, echo=False)
    items = pipeline.select(limit=limit)
    if not items:
        # "没有待处理条目"是【成功】，不是跳过，也不是失败。
        # 流水线完整地跑了一遍并得出了结论，这个结论就是"没有新东西"。
        return "没有待处理条目", None

    result, result_path = pipeline.run(items)
    summary = (
        f"处理 {len(result.outcomes)}/{result.candidates} 条，"
        f"成功 {result.ok_count}，格式问题 {result.format_error_count}，"
        f"调用失败 {result.call_failed_count}，花费 {result.total_cost} 元"
    )
    if result.truncated:
        summary += "（预算中断）"
    if result.call_failed_count and result.call_failed_count == len(result.outcomes):
        raise RuntimeError(
            f"所有条目都调用失败了：{summary}\n"
            f"  这通常意味着密钥、余额或网络出了问题，而不是内容有问题。"
        )
    return summary, result_path


def _step_digest(result_path: Path | None) -> str:
    from src.archive import DIGEST_DIR, REVIEW_DIR, load_result_set, render_digest, render_review
    from src.config import load_task

    if result_path is None:
        # 这一轮 pipeline 没产出新结果，于是没有可渲染的东西。
        #
        # 这里【绝不能】退回去渲染上一轮的结果文件：
        # 那会生成一份日期是今天、内容却是几天前的日报，而且它看起来完全正常。
        # 这是整条流水线里最危险的一个诱惑，因为它能让"什么都没发生"看起来像"一切正常"。
        #
        # 但也【不能】把它当成失败 —— 没有新内容本来就是正常的。
        # 所以它是跳过。
        raise StepSkipped("本轮没有新结果，没有可渲染的内容")

    records, summary = load_result_set(result_path)
    if not records:
        raise RuntimeError(f"结果文件是空的：{result_path.name}")

    task_cfg = load_task()
    stamp = result_path.stem
    DIGEST_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    digest_path = DIGEST_DIR / f"{stamp}.md"
    digest_path.write_text(
        render_digest(records, summary, stamp=stamp, task_cfg=task_cfg), encoding="utf-8"
    )
    review_path = REVIEW_DIR / f"{stamp}.txt"
    review_path.write_text(
        render_review(records, summary, stamp=stamp, task_cfg=task_cfg), encoding="utf-8"
    )
    ok = sum(1 for r in records if r.get("status") == "ok")
    return f"日报 {digest_path.name}（{ok} 条可用），审核表 out/review/{review_path.name}"


def build_steps(*, no_collect: bool, stub: bool, limit: int,
                provider: str | None, model: str | None) -> list[tuple[str, Callable[[], str]]]:
    """组装真实步骤。

    注意 pipeline 和 digest 之间怎么传那个结果文件路径：
    不是让 digest 自己去"找最新的一份"（那样它会找到上一轮的），
    而是把本轮实际产生的那个路径显式传过去。
    **隐式的"找最新的"是这类流水线最典型的静默错误来源。**
    """
    box: dict = {}

    def collect_step() -> str:
        return _step_collect(limit)

    def pipeline_step() -> str:
        summary, path = _step_pipeline(stub=stub, limit=limit, provider=provider, model=model)
        box["result_path"] = path
        return summary

    def digest_step() -> str:
        return _step_digest(box.get("result_path"))

    steps: list[tuple[str, Callable[[], str]]] = []
    if not no_collect:
        steps.append(("collect", collect_step))
    else:
        steps.append(("collect", lambda: "已跳过（--no-collect）"))
    steps.append(("pipeline", pipeline_step))
    steps.append(("digest", digest_step))
    return steps


# ------------------------------------------------------------------ 展示


def _short_ts(value: object) -> str:
    """把 ISO 时间压成 '10-11 21:01'。

    不是为了好看：完整 ISO 有 25 个字符，比列宽还长，
    于是它会和后面那一列挤在一起 —— 一张对齐坏掉的表，人是不会去读的。
    """
    if not value:
        return "（从未成功）"
    text = str(value)
    if "T" in text:
        date_part, _, rest = text.partition("T")
        return f"{date_part[5:]} {rest[:5]}"
    return text[:16]


def print_health(health: Health) -> int:
    rows = health.table()
    print()
    print("=" * 68)
    print("  任务健康表")
    print("=" * 68)

    if not rows:
        print()
        print("  还没有任何运行记录。跑一次：run.cmd daily --stub")
        print()
        return 0

    print()
    print(f"  {'任务':<12}{'连续失败':>8}  {'上次成功':<14}{'最近结果'}")
    print("  " + "-" * 64)
    for row in rows:
        streak = row["consecutive_failures"]
        mark = "!!" if streak >= health.escalate_after else ("? " if streak else "  ")
        summary = row["last_summary"] or row["last_error"] or ""
        print(f"  {mark}{row['task']:<10}{streak:>6}    {_short_ts(row['last_success']):<14}{str(summary)[:30]}")

    escalating = health.escalating()
    print()
    if escalating:
        print(f"  !! 有 {len(escalating)} 个任务连续失败达到 {health.escalate_after} 次：{'、'.join(escalating)}")
        print(f"     警报文件：{ALERT_PATH.relative_to(ROOT)}")
    else:
        print("  没有任务处于连续失败状态。")

    runs = health.data.get("runs", [])
    if runs:
        last = runs[-1]
        print()
        print(f"  最近一次运行 {last.get('ts')}　{'成功' if last.get('ok') else '失败'}"
              f"　用时 {last.get('seconds')}s")
        for step in last.get("steps", []):
            flag = "OK  " if step.get("ok") else "失败"
            detail = step.get("summary") or step.get("error") or ""
            print(f"    [{flag}] {step.get('name'):<10} {str(detail).splitlines()[0][:44]}")

    print()
    print("  「今天没新闻」和「流程三周前就死了」在表面上完全一样。")
    print("  这张表是你区分它们的唯一依据。")
    print()
    return 0


def print_outcome(outcome: dict) -> None:
    print()
    print("=" * 68)
    if outcome.get("ok"):
        extra = f"，跳过 {outcome['skipped']} 步" if outcome.get("skipped") else ""
        print(f"  本次运行完成{extra}　用时 {outcome['seconds']}s")
    else:
        print(f"  本次运行失败，停在步骤：{outcome.get('stopped_at')}　用时 {outcome['seconds']}s")
    print("=" * 68)
    for step in outcome["steps"]:
        if step.get("skipped"):
            flag, detail = "跳过", step.get("summary", "")
        elif step.get("ok"):
            flag, detail = "OK  ", step.get("summary", "")
        else:
            flag, detail = "失败", str(step.get("error", "")).splitlines()[0]
        print(f"  [{flag}] {step['name']:<10} {str(detail)[:52]}")
    if outcome.get("alert"):
        print()
        print(f"  !! 连续失败已达阈值，已写警报：{outcome['alert']}")
        print("     不要习惯一个长期存在的警报文件 —— 那等于没有警报。")
    if outcome.get("alert_cleared"):
        print()
        print("  之前的警报已自动撤销（有任务恢复正常了）。")
    print()
    return


# ------------------------------------------------------------------ 入口


def main() -> int:
    parser = argparse.ArgumentParser(description="权限与自动化：每日运行 + 健康记录")
    parser.add_argument("--daily", "-Daily", action="store_true", help="跑一次完整流程")
    parser.add_argument("--health", "-Health", action="store_true", help="看健康表（不联网、不要密钥）")
    parser.add_argument("--reset", "-Reset", default=None, help="清掉某个任务的失败计数")
    parser.add_argument("--no-collect", action="store_true", help="跳过 RSS 抓取")
    parser.add_argument("--stub", action="store_true", help="用假模型跑，不需要密钥")
    parser.add_argument("--continue-on-error", action="store_true", help="失败后继续执行后面的步骤")
    parser.add_argument("--limit", "-Limit", "-limit", type=int, default=0)
    parser.add_argument("--provider", "-Provider", default=None)
    parser.add_argument("--model", "-Model", default=None)
    args = parser.parse_args()

    policy = load_policy()
    health_cfg = policy.get("health", {})
    health = Health(
        escalate_after=int(health_cfg.get("escalate_after_consecutive_failures", DEFAULT_ESCALATE_AFTER)),
        keep_runs=int(health_cfg.get("keep_run_history", DEFAULT_KEEP_RUNS)),
    )

    if args.reset:
        if health.reset(args.reset):
            print(f"\n  已清掉 {args.reset} 的失败计数。\n")
            return 0
        print(f"\n  没有名为 {args.reset} 的任务记录。\n")
        return 1

    if args.health or not args.daily:
        return print_health(health)

    if args.stub:
        print()
        print("  " + "!" * 64)
        print("  --stub：流水线里用的是【假模型】。产出没有任何意义。")
        print("  " + "!" * 64)

    steps = build_steps(
        no_collect=args.no_collect, stub=args.stub,
        limit=args.limit, provider=args.provider, model=args.model,
    )
    outcome = run_daily(steps, health=health, continue_on_error=args.continue_on_error)
    print_outcome(outcome)

    if not outcome.get("ok"):
        print("  下一步：run.cmd health  看是哪个任务、连续失败了几次。")
        print()
        return 1

    digest_step = next((s for s in outcome["steps"] if s["name"] == "digest"), None)
    if digest_step is not None and digest_step.get("skipped"):
        # 这句提示是这一层存在的意义所在。
        # "什么都没有发生"必须有一个明确的回答，否则你无法区分它和"流程坏了"。
        print("  本轮没有新内容，日报没有重新生成 —— 这是正常的，不是故障。")
        print("  想确认系统还活着：run.cmd health")
    elif digest_step is not None:
        print("  下一步：打开 out/review/ 里那份审核表，改完跑 run.cmd review --apply <时间戳>")
    print()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[中断] 已完成的步骤记录已经落盘。\n")
        raise SystemExit(130)
