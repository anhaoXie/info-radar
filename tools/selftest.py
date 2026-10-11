"""零成本自检：不联网，不花钱，验证第 0 层和第 1 层的逻辑是对的。

【为什么值得有这么一个文件】

在你把密钥交给这套系统之前，你应该能确认它本身是好的。
"先跑一个不花钱的测试，再让它花钱" —— 这个习惯能省掉大量莫名其妙的调试。

它也演示了一个纪律：**能离线验证的逻辑，就不要用真金白银去验证。**
模型调用是有成本的，代码逻辑不是。把这两类错误分开排查。

用法：
    python tools/selftest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import task as task_mod  # noqa: E402
from src.config import load_policy, load_task  # noqa: E402
from src.ledger import Budget, BudgetExceeded, Ledger  # noqa: E402
from src.llm import FatalLLMError, LLMClient  # noqa: E402
from src.secrets import fingerprint, mask, _register  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [OK] {name}")
    else:
        FAILED.append(name)
        print(f"  [!!] {name}  {detail}")


def expect_raises(name: str, exc_type: type, fn, contains: str = "") -> None:
    try:
        fn()
    except exc_type as exc:
        if contains and contains not in str(exc):
            check(name, False, f"异常信息里没有 {contains!r}：{str(exc)[:120]}")
        else:
            check(name, True)
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"抛的是 {type(exc).__name__}，期望 {exc_type.__name__}")
    else:
        check(name, False, "没有抛出异常")


def main() -> int:
    print()
    print("=" * 68)
    print("  离线自检（不联网，不花费）")
    print("=" * 68)

    task_cfg = load_task()
    policy = load_policy()

    # ---------------------------------------------------------- ⑤ 脱敏
    print("\n[⑤ 权限层 · 脱敏]")
    _register("TEST_API_KEY", "sk-abcdef1234567890")
    check("已知密钥会被抹掉", "sk-abcdef1234567890" not in mask("我的密钥是 sk-abcdef1234567890 对吧"))
    check("指纹只留末 4 位", fingerprint("sk-abcdef1234567890") == "***7890")
    check("无关文本不被改动", mask("普通文本") == "普通文本")

    # ---------------------------------------------------------- ⓞ 解析
    print("\n[ⓞ 任务层 · 输出解析]")

    good = {"title": "标题", "summary": "摘要内容", "category": "产品发布",
            "importance": 3, "entities": ["甲公司"], "so_what": "影响", "evidence": ["原文一句"]}
    good_text = json.dumps(good, ensure_ascii=False)

    obj, err = task_mod.parse_json(good_text)
    check("能解析裸 JSON", obj == good and err is None, f"err={err}")

    obj, err = task_mod.parse_json(f"```json\n{good_text}\n```")
    check("能剥掉 markdown 围栏", obj == good and err is None, f"err={err}")

    obj, err = task_mod.parse_json(f"好的，以下是结果：\n{good_text}\n希望有帮助。")
    check("能从废话里抠出 JSON", obj == good and err is None, f"err={err}")

    obj, err = task_mod.parse_json('{"a": "} 括号在字符串里"}')
    check("字符串里的括号不会切错", obj == {"a": "} 括号在字符串里"}, f"err={err}")

    obj, err = task_mod.parse_json("这不是 JSON")
    check("非法输入返回错误而不是崩溃", obj is None and err is not None)

    obj, err = task_mod.parse_json("")
    check("空输出被识别", obj is None and err == "输出为空")

    # ---------------------------------------------------------- ⓞ 校验
    print("\n[ⓞ 任务层 · 结构校验]")

    problems, _ = task_mod.validate(task_cfg, dict(good))
    check("合法输出无硬伤", problems == [], f"problems={problems}")

    problems, _ = task_mod.validate(task_cfg, {**good, "category": "我自己发明的分类"})
    check("自创分类被判失败", any("category" in p for p in problems))

    problems, _ = task_mod.validate(task_cfg, {**good, "importance": 3.5})
    check("小数重要性被判失败", any("importance" in p for p in problems))

    problems, _ = task_mod.validate(task_cfg, {**good, "importance": 9})
    check("越界重要性被判失败", any("importance" in p for p in problems))

    problems, _ = task_mod.validate(task_cfg, {**good, "evidence": []})
    check("缺少依据被判失败", any("evidence" in p for p in problems))

    missing = {k: v for k, v in good.items() if k != "summary"}
    problems, _ = task_mod.validate(task_cfg, missing)
    check("缺字段被报出来", any("summary" in p for p in problems))

    problems, warnings = task_mod.validate(task_cfg, {**good, "title": "长" * 40})
    check("超长标题只是提醒，不算硬伤", problems == [] and any("title" in w for w in warnings))

    # ---------------------------------------------------------- 装配
    print("\n[ⓞ 任务层 · 消息装配]")

    doc = {"title": "T", "source": "S", "published": "2025-01-01", "body": "B"}
    messages = task_mod.build_messages(task_cfg, doc)
    check("基线是两条消息（system + user）", len(messages) == 2, f"实际 {len(messages)}")
    check("第一条是 system", messages[0]["role"] == "system")
    check("模板变量被替换", "{body}" not in messages[-1]["content"] and "B" in messages[-1]["content"])
    check("分类列表被注入", "产品发布" in messages[-1]["content"])

    with_shots = dict(task_cfg)
    with_shots["prompt"] = dict(task_cfg["prompt"])
    with_shots["prompt"]["few_shot"] = [{"input": doc, "output": good}]
    messages = task_mod.build_messages(with_shots, doc)
    check("加一条示例后变四条消息", len(messages) == 4, f"实际 {len(messages)}")
    check("示例的助手回复是 JSON", messages[2]["role"] == "assistant" and messages[2]["content"].startswith("{"))

    # ---------------------------------------------------------- ⑪ 评估集
    print("\n[⓪ 评估层 · 评估集读取]")

    from src.evaluate import load_eval_set

    rows, placeholder = load_eval_set(PROJECT_ROOT / "eval" / "eval_set.jsonl")
    check("能读出评估集", len(rows) >= 5, f"读到 {len(rows)} 条")
    check("能识别占位数据", placeholder is True)
    check("每条都有 gold.category", all(r.get("gold", {}).get("category") for r in rows))
    check("gold.category 都在允许列表里",
          all(r["gold"]["category"] in task_cfg["categories"] for r in rows),
          "有真值用了分类体系外的标签")

    # ---------------------------------------------------------- 账本
    print("\n[⓪⑤ 账本与预算]")

    with tempfile.TemporaryDirectory() as tmp:
        ledger = Ledger(policy, path=Path(tmp) / "ledger.jsonl")
        ledger.record("llm_call", provider="test", cost_cny=0.01, prompt_tokens=100, completion_tokens=50)
        ledger.record("llm_call", provider="test", cost_cny=0.02, prompt_tokens=200, completion_tokens=60)
        ledger.record("eval_run", total=2)

        rows = ledger.read_today()
        check("账本写进去能读回来", len(rows) == 3, f"读到 {len(rows)} 行")
        check("成本累加正确", abs(ledger.cost_today() - 0.03) < 1e-9, f"实际 {ledger.cost_today()}")

        usage = ledger.usage_today()
        check("只统计 llm_call 事件", usage["calls"] == 2, f"实际 {usage['calls']}")

        ledger.record("llm_call", leaked="sk-abcdef1234567890")
        raw = ledger.path.read_text(encoding="utf-8")
        check("账本落盘前会脱敏", "sk-abcdef1234567890" not in raw,
              "密钥被原样写进了账本 —— 这是最不能接受的失败")

        # 跨运行的每日上限：靠读回账本来判断，所以必须在临时账本还有效时测
        day_guard = Budget(
            {"budget": {"max_calls_per_run": 100, "max_cost_per_run_cny": 100,
                        "max_cost_per_day_cny": 0.02}},
            ledger=ledger,
        )
        expect_raises("跨运行的每日上限会生效", BudgetExceeded, day_guard.check, contains="今日")

    strict = {"budget": {"max_calls_per_run": 2, "max_cost_per_run_cny": 100, "max_cost_per_day_cny": 100}}
    budget = Budget(strict)
    budget.commit(cost=0.0, tokens=10)
    budget.commit(cost=0.0, tokens=10)
    expect_raises("超出调用次数上限会被拦住", BudgetExceeded, budget.check, contains="调用次数")

    tight = {"budget": {"max_calls_per_run": 100, "max_cost_per_run_cny": 0.05, "max_cost_per_day_cny": 100}}
    budget = Budget(tight)
    budget.commit(cost=0.04, tokens=10)
    try:
        budget.check()
        check("没超限额时不拦截", True)
    except BudgetExceeded:
        check("没超限额时不拦截", False, "花了 0.04 而限额 0.05，不该拦")

    budget.commit(cost=0.02, tokens=10)  # 累计 0.06，越过 0.05
    expect_raises("已超花费上限后，拦住下一次调用", BudgetExceeded, budget.check, contains="花费")

    proactive = Budget(tight)
    proactive.commit(cost=0.03, tokens=10)
    expect_raises(
        "给出预估花费时可以提前拦截",
        BudgetExceeded,
        lambda: proactive.check(projected_cost=0.05),
    )

    # ---------------------------------------------------------- 白名单
    print("\n[⑤ 权限层 · 白名单]")

    fake_providers = {
        "providers": {
            "evil": {
                "label": "未授权厂商",
                "base_url": "https://not-in-whitelist.example.com/v1",
                "api_key_env": "EVIL_API_KEY",
                "default_model": "whatever",
            }
        }
    }
    expect_raises(
        "未在白名单里的域名会被拒绝",
        FatalLLMError,
        lambda: LLMClient("evil", providers_cfg=fake_providers, policy_cfg=policy),
        contains="白名单",
    )

    unlisted = {"providers": {"bad": {
        "label": "没填模型", "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY", "default_model": None}}}
    expect_raises(
        "没配模型时给出可操作的指引",
        FatalLLMError,
        lambda: LLMClient("bad", providers_cfg=unlisted, policy_cfg=policy),
        contains="probe.py",
    )

    # ---------------------------------------------------------- 结果
    print()
    print("=" * 68)
    total = len(PASSED) + len(FAILED)
    if FAILED:
        print(f"  失败 {len(FAILED)} / {total}")
        for name in FAILED:
            print(f"    - {name}")
    else:
        print(f"  全部通过  {len(PASSED)} / {total}")
    print("=" * 68)
    print()
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
