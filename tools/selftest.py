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
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import make_eval_set  # noqa: E402
from src import collect as collect_mod  # noqa: E402
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

    # ---------------------------------------------------------- ① 数据层
    print("\n[① 数据层 · 源解析与归一]")

    rss_fixture = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>RSS item with content:encoded</title>
      <link>https://example.com/a?utm_source=x&amp;id=7</link>
      <pubDate>Mon, 06 Jan 2025 10:00:00 +0800</pubDate>
      <description>short</description>
      <content:encoded><![CDATA[<p>Full &amp; rich body text here.</p>]]></content:encoded>
    </item>
    <item>
      <title>RSS item with only description</title>
      <link>https://example.com/b</link>
      <description>Only a one-line summary.</description>
    </item>
  </channel>
</rss>"""

    atom_fixture = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Test</title>
  <entry>
    <title>Atom entry</title>
    <link rel="alternate" href="https://example.com/c"/>
    <updated>2025-01-07T12:00:00Z</updated>
    <summary>Summary text</summary>
  </entry>
</feed>"""

    rss_items = collect_mod.parse_feed(rss_fixture, "测试源")
    check("RSS 能解析出条目", len(rss_items) == 2, f"实际 {len(rss_items)}")
    check("content:encoded 优先于 description",
          "Full & rich body text here." in rss_items[0]["body"],
          f"实际 body={rss_items[0]['body']!r}")
    check("HTML 标签被去掉", "<p>" not in rss_items[0]["body"])
    check("链接里的 utm 追踪参数被剥掉",
          rss_items[0]["url"] == "https://example.com/a?id=7",
          f"实际 {rss_items[0]['url']}")
    check("pubDate 被取出", bool(rss_items[0]["published"]))
    check("只有 description 的条目也能解析",
          rss_items[1]["body"] == "Only a one-line summary.",
          f"实际 {rss_items[1]['body']!r}")

    atom_items = collect_mod.parse_feed(atom_fixture, "测试源")
    check("Atom 能解析出条目", len(atom_items) == 1, f"实际 {len(atom_items)}")
    check("Atom 的 link href 属性被取出",
          atom_items[0]["url"] == "https://example.com/c",
          f"实际 {atom_items[0]['url']}")
    check("Atom 的 summary 被取出", "Summary text" in atom_items[0]["body"])

    check("内容指纹忽略标点差异",
          collect_mod.content_hash("标题", "正文。") == collect_mod.content_hash("标题", "正文"),
          "标点不同但内容相同，指纹应该一样")
    check("不同正文的指纹不同",
          collect_mod.content_hash("标题", "正文A") != collect_mod.content_hash("标题", "正文B"))

    policy_now = load_policy()
    check("白名单外的新源会被拦下",
          collect_mod.missing_feed_hosts(
              [{"url": "https://new-source.example.com/feed"}], policy_now
          ) == ["new-source.example.com"])
    check("白名单内的源不会误报",
          collect_mod.missing_feed_hosts(
              [{"url": "https://www.qbitai.com/feed"}], policy_now
          ) == [])

    # ---------------------------------------------------------- 本地投喂
    print("\n[① 数据层 · 本地投喂]")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        md_path = tmp_dir / "doc.md"
        md_path.write_text("# 我的标题\n\n这是正文内容。\n", encoding="utf-8")
        parsed = collect_mod.read_inbox_file(md_path)
        check("Markdown 的 # 标题被取出", parsed is not None and parsed[0] == "我的标题", f"实际 {parsed}")
        check("Markdown 正文不含标题行", parsed is not None and "我的标题" not in parsed[1])

        html_path = tmp_dir / "doc.html"
        html_path.write_text(
            "<html><head><title>网页标题</title></head><body><p>段落一</p></body></html>",
            encoding="utf-8",
        )
        parsed = collect_mod.read_inbox_file(html_path)
        check("HTML 的 title 被取出", parsed is not None and parsed[0] == "网页标题", f"实际 {parsed}")
        check("HTML 标签被去掉", parsed is not None and "<p>" not in parsed[1])

        json_path = tmp_dir / "doc.json"
        json_path.write_text(json.dumps({"title": "JSON标题", "body": "JSON正文"}), encoding="utf-8")
        check("JSON 的 title/body 被取出",
              collect_mod.read_inbox_file(json_path) == ("JSON标题", "JSON正文"))

        empty_path = tmp_dir / "empty.txt"
        empty_path.write_text("   \n\n", encoding="utf-8")
        check("空文件返回 None 而不是空字符串", collect_mod.read_inbox_file(empty_path) is None)

        # 标题猜测的三条规则。第 3 条是后补的，起因是一个真实的坏结果：
        # 只有一段长话的 .txt，原来会把第一行截断成标题，于是标题栏里出现
        # 一句没头没尾的长句，人完全看不出这篇讲什么。
        short_path = tmp_dir / "short.txt"
        short_path.write_text("这是一个短标题\n正文第一段。\n", encoding="utf-8")
        parsed = collect_mod.read_inbox_file(short_path)
        check("短的第一行会被当成标题", parsed is not None and parsed[0] == "这是一个短标题",
              f"实际 {parsed[0] if parsed else None}")
        check("标题行不会重复出现在正文里",
              parsed is not None and "这是一个短标题" not in parsed[1], f"实际 {parsed[1] if parsed else None}")

        long_path = tmp_dir / "longfirstline.txt"
        long_path.write_text(
            "这是一段很长的正文，它并不像一个标题，因为它的长度超过了一般标题的范围，而且读起来是完整的句子。\n第二行内容。",
            encoding="utf-8",
        )
        parsed = collect_mod.read_inbox_file(long_path)
        check("以句号结尾的第一行（完整句子）不会被误当成标题",
              parsed is not None and parsed[0] == "longfirstline",
              f"猜不出来时应退回文件名，实际 {parsed[0] if parsed else None}")
        check("退回文件名时正文保留全文",
              parsed is not None and "这是一段很长的正文" in parsed[1],
              f"实际正文 {parsed[1][:40] if parsed else None}")

        # 正交特征：长度不够就换"结尾标点"来判，比反复调阈值有效
        punct_path = tmp_dir / "punct.title.txt"
        punct_path.write_text("短但以句号结尾的一句话。\n下面还有正文。\n", encoding="utf-8")
        parsed = collect_mod.read_inbox_file(punct_path)
        check("短但以句号结尾的行也不算标题",
              parsed is not None and parsed[0] == "punct.title",
              f"实际 {parsed[0] if parsed else None}")

        longtitle_path = tmp_dir / "longtitle.txt"
        longtitle_path.write_text(
            "这是一个相当长的标题它没有以句号结尾所以它应该被认出来\n正文在这里。\n",
            encoding="utf-8",
        )
        parsed = collect_mod.read_inbox_file(longtitle_path)
        check("长但不以句号结尾的行仍会被当成标题",
              parsed is not None and parsed[0].startswith("这是一个相当长的标题"),
              f"实际 {parsed[0] if parsed else None}")

        single_path = tmp_dir / "singleline.txt"
        single_path.write_text("只有一行，没有第二行可以当正文。", encoding="utf-8")
        parsed = collect_mod.read_inbox_file(single_path)
        check("只有一行时不丢内容",
              parsed is not None and "只有一行" in parsed[1], f"实际 {parsed}")

    # ---------------------------------------------------------- 标注
    print("\n[⓪ 评估层 · 标注解析与抽样]")

    with tempfile.TemporaryDirectory() as tmp:
        labels_path = Path(tmp) / "labels.txt"
        labels_path.write_text(
            "# 注释行会被忽略\n"
            "\n"
            "aaa111 | 模型与技术 | 4\n"
            "bbb222 | ? | ?\n"
            "ccc333 | 产品发布 | 99\n"
            "ddd444 |  | \n",
            encoding="utf-8",
        )
        labels = make_eval_set.parse_labels(labels_path)
        check("能解析正常标注", labels.get("aaa111") == ("模型与技术", 4, False), f"实际 {labels.get('aaa111')}")
        check("问号被识别为不确定", labels.get("bbb222") == (None, None, True), f"实际 {labels.get('bbb222')}")
        check("注释行与空行被忽略", "" not in labels)
        check("越界的重要性先读进来，交给后面的校验去拦",
              labels.get("ccc333") == ("产品发布", 99, False), f"实际 {labels.get('ccc333')}")
        check("留空的行被识别为未填", labels.get("ddd444") == (None, None, False), f"实际 {labels.get('ddd444')}")

    fake_items = (
        [{"id": f"a{i}", "source": "源A", "body_chars": 1000 - i} for i in range(10)]
        + [{"id": f"b{i}", "source": "源B", "body_chars": 900 - i} for i in range(10)]
    )
    picked = make_eval_set.pick_diverse(fake_items, 6, per_source_cap=6)
    sources_used = Counter(item["source"] for item in picked)
    check("抽样会在来源之间轮流", len(sources_used) == 2, f"实际只用到 {list(sources_used)}")
    check("抽样遵守单源上限", all(v <= 6 for v in sources_used.values()))
    check("抽样数量正确", len(picked) == 6, f"实际 {len(picked)}")

    capped = make_eval_set.pick_diverse(fake_items, 20, per_source_cap=3)
    check("单源上限卡死时不会死循环", len(capped) == 6, f"实际 {len(capped)}")

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
