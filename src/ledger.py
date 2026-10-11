"""第 ⓪ / ⑤ 层 · 账本（ledger）与预算（budget）

【为什么在写业务逻辑之前先写这个】

"这次改动到底变好了没有"这个问题，只有两种回答方式：

    靠感觉：  好像准了一点。
    靠账本：  分类准确率 55% -> 80%，单条成本 0.31 分 -> 0.42 分，延迟 2.1s -> 3.4s。

账本就是把第二种回答变成可能的那个东西。等出了事再补日志，你会失去出事那段时间的全部数据。

【账本是什么】

    out/logs/ledger-YYYY-MM-DD.jsonl
    一行一个事件，追加写，历史行永不修改。
    按天分文件，方便按天算成本，也方便直接删掉旧数据。

【预算是什么】

    config/policy.json 里的数字不是文档，是下面这个 raise。
    写在文档里的限额等于没有限额 —— 因为没人在运行时读文档。
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .secrets import PROJECT_ROOT, mask

LOG_DIR = PROJECT_ROOT / "out" / "logs"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class BudgetExceeded(RuntimeError):
    """超出 policy.json 允许的花费或调用次数。"""


class Budget:
    """本次运行的预算闸门。

    这是"权限"从口号变成代码的第二个地方（第一个是密钥）。
    它的价值不在于省钱，而在于：**一个会失控花钱的流程，你不敢多跑，也就不敢改进它。**

    【一个重要且必须说清的语义：这个闸门是"下一次调用前"生效的】

    调用次数是事先就知道的，所以能在调用【前】精确拦住。
    花费是调用【后】才知道的，所以只能在调用【前】检查"到目前为止是否已经超了"。

    代价：最多会超出一次调用的量，然后立刻停住。
    这个代价是可以接受的 —— 它保证的是一件更重要的事：
    **流程会停下来，而不是无限跑下去。** 一个跑一夜的脚本刷掉几百块，
    原因从来不是"少拦了一次"，而是"根本没有闸门"。
    """

    def __init__(self, policy: dict, *, ledger: "Ledger | None" = None) -> None:
        budget = (policy or {}).get("budget", {})
        self.max_calls = int(budget.get("max_calls_per_run", 10**9))
        self.max_cost_run = float(budget.get("max_cost_per_run_cny", float("inf")))
        self.max_cost_day = float(budget.get("max_cost_per_day_cny", float("inf")))
        self.calls = 0
        self.cost = 0.0
        self.tokens = 0
        self._ledger = ledger
        self._lock = threading.Lock()

    def check(self, *, projected_cost: float = 0.0) -> None:
        """在真正发请求【之前】调用。宁可提前拦住，不要事后后悔。

        projected_cost 是"这次大概要花多少"的估计值。估不出来就传 0，
        闸门仍然会在累计花费已经越线时拦住下一次调用。
        """
        with self._lock:
            if self.calls + 1 > self.max_calls:
                raise BudgetExceeded(
                    f"本次运行调用次数将超过上限：已 {self.calls} 次，上限 {self.max_calls} 次。\n"
                    f"  改 config/policy.json 里的 budget.max_calls_per_run，或减少本次处理量。"
                )
            if self.cost + projected_cost > self.max_cost_run:
                raise BudgetExceeded(
                    f"本次运行花费已超过上限：已花 {self.cost:.4f} 元（含预计 {projected_cost:.4f} 元）"
                    f"> 上限 {self.max_cost_run:.4f} 元。\n"
                    f"  改 config/policy.json 里的 budget.max_cost_per_run_cny，或减少本次处理量。"
                )

        if self._ledger is not None and self.max_cost_day != float("inf"):
            spent = self._ledger.cost_today()
            if spent + projected_cost > self.max_cost_day:
                raise BudgetExceeded(
                    f"今日花费将超过上限：今日已用 {spent:.4f} 元 + 预计 {projected_cost:.4f} 元 "
                    f"> 上限 {self.max_cost_day:.4f} 元。\n"
                    f"  这是跨运行的保护，账本里记着呢。"
                )

    def commit(self, *, cost: float, tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.cost += float(cost or 0.0)
            self.tokens += int(tokens or 0)

    @property
    def summary(self) -> dict:
        return {"calls": self.calls, "cost_cny": round(self.cost, 6), "tokens": self.tokens}


class Ledger:
    """追加写的事件账本。"""

    def __init__(self, policy: dict | None = None, *, path: Path | None = None) -> None:
        self.policy = policy or {}
        self._explicit_path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        if self._explicit_path is not None:
            return self._explicit_path
        day = datetime.now().strftime("%Y-%m-%d")
        return LOG_DIR / f"ledger-{day}.jsonl"

    def record(self, event: str, **fields) -> None:
        """写一行事件。所有字符串字段都会被脱敏，然后才落盘。"""
        entry = {"ts": _now_iso(), "event": event}
        for key, value in fields.items():
            entry[key] = mask(value) if isinstance(value, str) else value

        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def read_today(self) -> list[dict]:
        """读回今天的账本。预算判断和自检都靠它。"""
        if not self.path.exists():
            return []
        rows: list[dict] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 半行写入（比如断电）：跳过，不要让账本本身把程序搞挂
        return rows

    def cost_today(self) -> float:
        return sum(float(r.get("cost_cny") or 0.0) for r in self.read_today() if r.get("event") == "llm_call")

    def usage_today(self) -> dict:
        rows = [r for r in self.read_today() if r.get("event") == "llm_call"]
        return {
            "calls": len(rows),
            "cost_cny": round(sum(float(r.get("cost_cny") or 0.0) for r in rows), 6),
            "prompt_tokens": sum(int(r.get("prompt_tokens") or 0) for r in rows),
            "completion_tokens": sum(int(r.get("completion_tokens") or 0) for r in rows),
        }
