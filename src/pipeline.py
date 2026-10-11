"""第 ③ 层 · 工作流层：编排、失败隔离、断点续跑

【这一层负责什么】

不是"调用模型"（那是 ② 层），也不是"抓数据"（那是 ① 层）。
它负责的是：**把流水线的步骤串起来，并决定每一步出问题时怎么办。**

具体三件事：

    1. 编排     哪些条目要处理、按什么顺序、每一步的输入输出是什么
    2. 失败隔离  一条坏掉不能拖垮整批
    3. 断点续跑  中途挂了，重跑时不重复已完成的工作（也就不重复花钱）

【为什么"断点续跑"值得单独拿出来讲】

因为它决定了一件很实际的事：**你敢不敢改进这套系统。**

一个会重复花钱的流程，你不敢多跑；跑得越少，你对它的信任越少；
到最后你会退回"凭感觉看一眼"。
所以断点续跑不是性能优化，是让"迭代"这件事成为可能的前提。

【一个必须说清的决策：失败的条目要不要标记为已处理】

    标记    -> 不会每次重跑都重试，不会静默烧钱。但失败容易被遗忘。
    不标记  -> 失败一定会被重试到成功。但一个永久损坏的条目会每次都在烧钱。

这里选择【标记，并记录失败原因】，另外提供 --retry-failed 让你**主动**重试。

理由：默认重试是错的 —— 它会把"一次偶发失败"变成"每次运行都扣钱"，
并且掩盖问题。但必须让失败**可见**、让重试**可用**。
把"重试"从默认行为变成一次有意识的动作，是这个设计的全部要点。

【算与存分离】

run_pipeline() 是纯编排：它不碰磁盘、不碰索引，只接收条目、返回结果。
磁盘读写由 Pipeline 这个壳负责，并且**每条处理完立刻落盘**。

这样做有两个好处：
    可以测试 —— mocktest 能用假服务端把整条流水线跑一遍，不花钱
    可以续跑 —— 第 7 条崩了，前 6 条的标记已经在磁盘上了

用法：
    python src/pipeline.py                    # 处理所有待处理条目
    python src/pipeline.py --limit 5          # 只处理 5 条
    python src/pipeline.py --stub             # 用假模型跑，验证链路，不花钱
    python src/pipeline.py --retry-failed     # 重试之前失败的条目
    python src/pipeline.py --dry-run          # 只看要处理哪些，不发请求
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import task as task_mod  # noqa: E402
from src.config import load_policy, load_task, main_guard  # noqa: E402
from src.ledger import Budget, BudgetExceeded, Ledger  # noqa: E402
from src.llm import LLMClient, LLMError, LLMResult  # noqa: E402
from src.secrets import PROJECT_ROOT as ROOT  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
INDEX_PATH = ROOT / "data" / "cache" / "index.json"
RESULTS_DIR = ROOT / "out" / "results"


# ------------------------------------------------------------------ 结果结构


@dataclass
class ItemOutcome:
    """一条条目的处理结果。它是 ③ 层交给 ④ 层的东西。"""

    item_id: str
    title: str = ""
    source: str = ""
    status: str = "pending"  # ok | format_error | call_failed
    output: dict | None = None
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    latency_ms: int = 0
    cost_cny: float = 0.0
    attempts: int = 1

    @property
    def succeeded(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict:
        return {
            "id": self.item_id,
            "title": self.title,
            "source": self.source,
            "status": self.status,
            "output": self.output,
            "problems": self.problems,
            "warnings": self.warnings,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "cost_cny": self.cost_cny,
            "attempts": self.attempts,
        }


@dataclass
class RunResult:
    outcomes: list[ItemOutcome] = field(default_factory=list)
    candidates: int = 0
    truncated: bool = False
    stop_reason: str = ""

    @property
    def ok_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "ok")

    @property
    def format_error_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "format_error")

    @property
    def call_failed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "call_failed")

    @property
    def total_cost(self) -> float:
        return round(sum(o.cost_cny for o in self.outcomes), 6)


# ------------------------------------------------------------------ 假模型


class StubClient:
    """没有密钥时用的假模型。

    【它不是 mock，它是"空转"】

    mocktest 里的假服务端是用来【验证客户端】的；
    这个 StubClient 是用来让你在还没有密钥时【看清工作流长什么样】的。

    它的产出【没有意义】，所以它说的每句话都带着"[假模型]"前缀。
    一个看起来像真结果的假结果，比一个明显的错误危险得多。
    """

    provider_name = "stub"
    model = "stub-model"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, *, purpose: str = "", **kwargs) -> LLMResult:
        self.calls += 1
        user_text = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                user_text = str(message.get("content", ""))
                break
        head = user_text.split("【正文】", 1)[-1].strip().replace("\n", " ")[:24]

        payload = {
            "title": f"[假模型] {head}",
            "summary": "这是假模型生成的占位摘要。它的唯一作用是让你在没有密钥时看到完整链路跑通，"
                       "内容本身没有任何意义，不要用它做任何判断。",
            "category": "其他",
            "importance": 3,
            "entities": [],
            "so_what": "[假模型] 无实际含义。",
            "evidence": ["[假模型] 这条不是原文摘录。"],
        }
        return LLMResult(
            text=json.dumps(payload, ensure_ascii=False),
            provider=self.provider_name,
            model=self.model,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            latency_ms=1,
            cost_cny=0.0,
            cost_known=False,
            attempts=1,
            request_id=f"stub-{self.calls}",
        )


# ------------------------------------------------------------------ 纯编排


def run_pipeline(
    client,
    task_cfg: dict,
    items: list[dict],
    *,
    params: dict | None = None,
    echo: bool = True,
    on_result: Callable[[ItemOutcome], None] | None = None,
) -> RunResult:
    """把一批条目跑完。不碰磁盘 —— 落盘由调用方通过 on_result 负责。

    返回 RunResult；其中 truncated=True 表示因预算耗尽而提前中止。
    """
    params = params if params is not None else task_cfg.get("prompt", {}).get("params", {})
    result = RunResult(candidates=len(items))
    total = len(items)

    for index, item in enumerate(items, 1):
        item_id = str(item.get("id") or f"#{index}")
        title = str(item.get("title") or "")

        if echo:
            print(f"[{index:>3}/{total}] {item_id:<14} {title[:36]}")

        outcome = ItemOutcome(item_id=item_id, title=title, source=str(item.get("source") or ""))

        try:
            messages = task_mod.build_messages(task_cfg, item)
            completion = client.chat(
                messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                json_mode=bool(params.get("json_mode")),
                purpose=f"pipeline:{item_id}",
            )
        except BudgetExceeded as exc:
            # 预算用尽：中止，但已经处理完的条目标记必须已经在磁盘上。
            # 这里不产出 outcome —— 这一条根本没被处理过，不该被标记为已处理。
            if echo:
                print()
                print("  " + "-" * 64)
                print(f"  预算用尽，中止：{exc}")
                print(f"  已完成的 {len(result.outcomes)} 条会保留；未处理的条目下次继续。")
                print("  " + "-" * 64)
            result.truncated = True
            result.stop_reason = "budget"
            break
        except LLMError as exc:
            # 失败隔离：一条失败不能让整批停下。
            outcome.status = "call_failed"
            outcome.error = str(exc)
            result.outcomes.append(outcome)
            if on_result:
                on_result(outcome)
            if echo:
                print(f"          调用失败：{str(exc)[:100]}")
            continue

        parsed, parse_error = task_mod.parse_json(completion.text)
        if parsed is None:
            problems, warnings = [parse_error or "解析失败"], []
        else:
            problems, warnings = task_mod.validate(task_cfg, parsed)

        outcome.output = parsed
        outcome.problems = problems
        outcome.warnings = warnings
        outcome.status = "ok" if not problems else "format_error"
        outcome.latency_ms = completion.latency_ms
        outcome.cost_cny = completion.cost_cny
        outcome.attempts = completion.attempts

        result.outcomes.append(outcome)
        if on_result:
            on_result(outcome)

        if echo:
            if outcome.status == "ok":
                category = (parsed or {}).get("category", "?")
                importance = (parsed or {}).get("importance", "?")
                print(f"          [OK] {completion.latency_ms}ms  分类={category}  重要性={importance}")
            else:
                print(f"          [格式] {completion.latency_ms}ms")
            for problem in problems:
                print(f"            ! {problem}")

    return result


# ------------------------------------------------------------------ 壳：磁盘


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


class Pipeline:
    """把编排（run_pipeline）和磁盘（索引、原文、结果）接起来。

    每处理完一条就立刻更新索引并落盘 —— 这是"断点续跑"的全部实现。
    如果等到整批跑完才写，那么第 199 条崩溃会让前 198 条的记录一起消失，
    而它们是花过钱的。
    """

    def __init__(
        self,
        client,
        task_cfg: dict,
        *,
        raw_dir: Path | None = None,
        index_path: Path | None = None,
        results_dir: Path | None = None,
        echo: bool = True,
    ) -> None:
        self.client = client
        self.task_cfg = task_cfg
        self.raw_dir = raw_dir or RAW_DIR
        self.index_path = index_path or INDEX_PATH
        self.results_dir = results_dir or RESULTS_DIR
        self.echo = echo

        self.index = _load_json(self.index_path, {"items": {}, "content_hashes": {}})
        self.index.setdefault("items", {})
        self.results: list[ItemOutcome] = []

    # ---------------------------------------------------------- 选择

    def select(self, *, limit: int = 0, sources: list[str] | None = None, retry_failed: bool = False) -> list[dict]:
        """挑出待处理的条目，并把原文读进来。

        排序：新抓到的在前。对资讯类任务，越新越值得先处理；
        而对于跑批，一个确定的顺序比"最优顺序"更重要 —— 它让结果可复现。
        """
        pending: list[dict] = []
        for item_id, meta in self.index["items"].items():
            processed = bool(meta.get("processed"))
            if processed and not (retry_failed and meta.get("status") != "ok"):
                continue
            if sources and meta.get("source") not in sources:
                continue
            record = self._read_raw(item_id)
            if record is None:
                continue
            pending.append(record)

        pending.sort(key=lambda r: str(r.get("fetched_at") or ""), reverse=True)
        return pending[:limit] if limit else pending

    def is_retry(self, item_id: str) -> bool:
        """这条是不是「处理过但失败了」—— 也就是 --retry-failed 额外捞回来的那些。

        单独给它一个名字，是因为这件事很容易被误解：
        用户看到 --retry-failed，会以为"只跑失败的那几条"。
        而实际行为是"把失败的也捞回来，和没跑过的混在一起处理"。

        在花掉真金白银之前，这个区别必须说清楚：
        你以为要处理 2 条，实际处理 116 条 —— 差 50 倍的钱。
        """
        meta = self.index["items"].get(item_id) or {}
        return bool(meta.get("processed")) and meta.get("status") != "ok"

    def _read_raw(self, item_id: str) -> dict | None:
        path = self.raw_dir / f"{item_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # ---------------------------------------------------------- 落盘

    def _mark(self, outcome: ItemOutcome) -> None:
        """把一条结果写进索引并立刻落盘。"""
        meta = self.index["items"].setdefault(outcome.item_id, {})
        meta["processed"] = True
        meta["processed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        meta["status"] = outcome.status
        meta["attempts"] = int(meta.get("attempts") or 0) + 1
        meta["last_error"] = outcome.error
        self.results.append(outcome)
        _atomic_write_json(self.index_path, self.index)

    # ---------------------------------------------------------- 跑

    def run(self, items: list[dict]) -> tuple[RunResult, Path | None]:
        result = run_pipeline(
            self.client,
            self.task_cfg,
            items,
            echo=self.echo,
            on_result=self._mark,
        )

        # 预算是中途断的，那些没被处理过的条目标记不能丢原样 ——
        # 它们仍然 processed=False，下次运行会自动接着跑。
        result_path = self._write_results(result) if self.results else None
        return result, result_path

    def _write_results(self, result: RunResult) -> Path:
        """把这一轮的产出写成 JSONL —— ③ 层交给 ④ 层的中间结果。

        注意它是【机器格式】。给人看的日报是 ④ 层的事。
        这个分工不是洁癖：中间结果要能被程序继续处理（归档、审核、回流），
        而人看的东西要能被眼睛扫。把两者混在一个文件里，两边都会做不好。
        """
        self.results_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.results_dir / f"{stamp}.jsonl"

        lines = []
        for outcome in result.outcomes:
            lines.append(json.dumps(outcome.as_dict(), ensure_ascii=False))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        summary = {
            "ts": stamp,
            "provider": getattr(self.client, "provider_name", "?"),
            "model": getattr(self.client, "model", "?"),
            "candidates": result.candidates,
            "processed": len(result.outcomes),
            "ok": result.ok_count,
            "format_error": result.format_error_count,
            "call_failed": result.call_failed_count,
            "total_cost_cny": result.total_cost,
            "truncated": result.truncated,
            "stop_reason": result.stop_reason,
        }
        _atomic_write_json(self.results_dir / f"{stamp}.summary.json", summary)
        return path


# ------------------------------------------------------------------ 入口


def _print_report(result: RunResult, result_path: Path | None, elapsed: float) -> None:
    print()
    print("=" * 68)
    if result.truncated:
        print("  本轮被预算中断（未处理的条目下次会接着跑）")
    else:
        print("  本轮完成")
    print("=" * 68)
    print(f"  候选条目   {result.candidates}")
    print(f"  已处理     {len(result.outcomes)}")
    print(f"  成功       {result.ok_count}")
    print(f"  格式不合规 {result.format_error_count}")
    print(f"  调用失败   {result.call_failed_count}")
    print(f"  本轮花费   {result.total_cost} 元")
    print(f"  用时       {elapsed:.1f} 秒")
    if result_path:
        print(f"  产出       {result_path.relative_to(ROOT)}")
    else:
        print("  产出       （这一轮没有处理任何条目）")

    remaining = result.candidates - len(result.outcomes)
    if remaining > 0:
        print()
        print(f"  还有 {remaining} 条没处理。再跑一次 run.cmd pipeline 就会接着处理 ——")
        print("  已完成的不会重做，所以不会重复花钱。这就是断点续跑。")

    if result.format_error_count:
        print()
        print("  格式不合规的条目值得看一眼：是提示词的问题，还是这个任务本身有歧义？")
        print("  它们在数据里是完整的，不会丢。")

    print()
    return


def main() -> int:
    parser = argparse.ArgumentParser(description="工作流层：编排、失败隔离、断点续跑")
    parser.add_argument("--limit", "-Limit", "-limit", type=int, default=0, help="最多处理 N 条")
    parser.add_argument("--source", "-Source", default=None, help="只处理某个来源")
    parser.add_argument("--stub", action="store_true", help="用假模型跑，验证链路，不花钱")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="把【之前处理失败的】条目也捞回来一起处理（注意：不是只重试它们，"
             "没跑过的照样会跑）",
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出要处理哪些，不发请求")
    parser.add_argument("--provider", "-Provider", default=None)
    parser.add_argument("--model", "-Model", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    task_cfg = load_task()
    policy = load_policy()

    # ---------------------------------------------------------------- 选条目
    # 这一步只读磁盘：不碰网络，不需要密钥，不花钱。
    pipeline = Pipeline(None, task_cfg, echo=not args.quiet)  # type: ignore[arg-type]
    items = pipeline.select(
        limit=args.limit,
        sources=[args.source] if args.source else None,
        retry_failed=args.retry_failed,
    )

    if not items:
        print()
        print("  没有待处理的条目。")
        print()
        print("  可能的原因：")
        print("    - 还没抓数据：run.cmd collect 或 run.cmd inbox")
        print("    - 全都处理过了：想重跑失败的加 --retry-failed")
        print()
        return 0

    if args.dry_run:
        # 一个"只看不动"的命令，不该强迫你先交出密钥。
        # 这条顺序是有意安排的：把所有只读的检查放在需要凭证的操作之前。
        # 一个需要密钥才能告诉你"我打算做什么"的工具，你只会在出错时才用它。
        print()
        print(f"  待处理 {len(items)} 条（--dry-run：不发请求、不花钱）")
        if args.retry_failed:
            _retried = [it for it in items if pipeline.is_retry(it["id"])]
            if _retried:
                print(f"  其中 {len(_retried)} 条是【之前失败的】，会一并重试")
                print(f"  （剩下 {len(items) - len(_retried)} 条是从来没跑过的）")
                print()
                print("  ! --retry-failed 不是「只重试失败的」——"
                      "它把失败的也捞回来，和你没跑过的混在一起。")
                print("    想只跑那几条，把它们的 id 记下来，先 --source 或分批处理。")
        print()
        for index, item in enumerate(items, 1):
            print(f"  {index:>3}. {item['id']:<14} {str(item.get('title'))[:44]}")
        print()
        print("  去掉 --dry-run 才会真正处理。")
        print()
        return 0

    # ---------------------------------------------------------------- 建客户端
    # 到这一步才需要模型。
    ledger = Ledger(policy)
    budget = Budget(policy, ledger=ledger)

    if args.stub:
        client = StubClient()
        print()
        print("  " + "!" * 64)
        print("  --stub：正在用一个【假模型】跑。产出没有任何意义。")
        print("  它的作用是让你在没有密钥时看清工作流长什么样，以及验证链路是通的。")
        print("  真跑请去掉 --stub。")
        print("  " + "!" * 64)
    else:
        try:
            client = LLMClient(
                args.provider, model=args.model, policy_cfg=policy, ledger=ledger, budget=budget
            )
        except (LLMError, RuntimeError) as exc:
            print(f"\n[错误] {exc}\n")
            return 2

    pipeline.client = client

    print()
    print(f"  待处理 {len(items)} 条   厂商/模型 {getattr(client, 'provider_name', '?')} / {getattr(client, 'model', '?')}")

    started = time.perf_counter()
    result, result_path = pipeline.run(items)
    _print_report(result, result_path, time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main_guard(main)())
    except KeyboardInterrupt:
        print("\n[中断] 已处理完的条目标记已经落盘，重跑不会重复花钱。\n")
        raise SystemExit(130)
