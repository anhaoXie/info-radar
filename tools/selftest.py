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

    # 填密钥时的两种常见复制错误，都只在调用时表现为"密钥被拒绝"，
    # 于是人会回去反复重新申请 —— 而问题是复制多带了几个字。
    from src.secrets import key_problems

    check("正常的密钥不会被误报", key_problems("sk-abcdef1234567890abcdef") == [],
          f"实际 {key_problems('sk-abcdef1234567890abcdef')}")
    check("带空格会被指出", any("空格" in p for p in key_problems("sk-abc def1234567890")))
    check("带换行会被指出", any("空格" in p for p in key_problems("sk-abcdef1234567890\n")))
    check("太短会被指出", any("字符" in p for p in key_problems("sk-123")))
    check("把「API Key:」也复制进来会被指出",
          any("冒号" in p for p in key_problems("API Key: sk-abcdef1234567890")))
    check("误把请求头写法当密钥会被指出",
          any("bearer" in p.lower() for p in key_problems("Bearer sk-abcdef1234567890")))
    check("空值会被指出", key_problems("") == ["是空的"])

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

    # 【不要拿真实的 eval/eval_set.jsonl 去断言内容】
    #
    # 第一版这几条断言写的是"真实评估集里含占位数据、每条都有 gold.category"。
    # 那时它是对的 —— 因为评估集还是占位版。
    # 可等你哪天 compile 出自己的评估集（那是【正确用法】），
    # 这几条就会开始失败，然后你跑 selftest 会看到一片红，以为系统坏了。
    #
    # 一个"用户正确使用它就会失败"的测试，比没有测试更糟。
    # 所以：行为用临时样本测，真实文件只做"读得动、形状对"的宽松检查。
    with tempfile.TemporaryDirectory() as tmp:
        fixture_set = Path(tmp) / "eval_set.jsonl"
        fixture_set.write_text(
            '{"id":"p1","_placeholder":true,"gold":{"category":"其他","importance":3}}\n'
            '{"id":"p2","gold":{"category":"产品发布","importance":4}}\n',
            encoding="utf-8",
        )
        fixture_rows, fixture_placeholder = load_eval_set(fixture_set)
    check("能读出评估集", len(fixture_rows) == 2, f"读到 {len(fixture_rows)} 条")
    check("能识别占位数据", fixture_placeholder is True)
    check("没有占位标记时不会被误判成占位", fixture_placeholder and fixture_rows[1]["id"] == "p2")

    real_set = PROJECT_ROOT / "eval" / "eval_set.jsonl"
    if real_set.exists():
        try:
            real_rows, _ = load_eval_set(real_set)
            bogus = [
                r.get("id") for r in real_rows
                if r.get("gold", {}).get("category")
                and r["gold"]["category"] not in task_cfg["categories"]
            ]
            check("真实的评估集能被读出来（不管它是占位版还是你自己标的）",
                  True, "")
            check("真实评估集里的分类都在允许列表内", not bogus, f"越界：{bogus}")
        except SystemExit as exc:
            check("真实的评估集能被读出来", False, str(exc)[:110])

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
    # 从配置里现取一个真实存在的域名，不要硬编码某个具体的源 ——
    # 硬编码的断言会在你换掉源列表的那一刻失效，
    # 而它失败的原因（"白名单里的源被判为不在白名单里"）看起来像是代码坏了。
    feed_hosts = [h for h in ((policy_now.get("network") or {}).get("feed_hosts") or [])
                  if isinstance(h, str) and "." in h]
    check("白名单里确实有信息源域名（配置不是空的）", bool(feed_hosts),
          "feed_hosts 是空的，后面的白名单测试没有意义")
    check("白名单外的新源会被拦下",
          collect_mod.missing_feed_hosts(
              [{"url": "https://definitely-not-listed.example.com/feed"}], policy_now
          ) == ["definitely-not-listed.example.com"])
    check("白名单内的源不会误报",
          bool(feed_hosts)
          and collect_mod.missing_feed_hosts(
              [{"url": f"https://{feed_hosts[0]}/feed"}], policy_now
          ) == [],
          f"实际用 {feed_hosts[0] if feed_hosts else '(空)'} 测的")

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

    # 速览版工作纸：标注这件事的瓶颈不是"读得不够仔细"，是"根本没人愿意花半小时"。
    # 速览版存在的唯一目的是让它真的发生，所以这两条断言守的是"能不能用"，
    # 不是"内容全不全"。
    lite_fixture = [
        {
            "id": f"L{i:02d}",
            "title": f"第 {i} 条的标题",
            "source": "测试源",
            "body_chars": 3000,
            "body": f"第 {i} 条的开头一段。这里是用来验证摘录有没有被截断的填充文字。" * 20,
        }
        for i in range(1, 6)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        real_lite = make_eval_set.LITE_PATH
        make_eval_set.LITE_PATH = Path(tmp) / "worksheet-lite.md"
        try:
            make_eval_set.write_worksheet_lite(lite_fixture, task_cfg)
            lite = make_eval_set.LITE_PATH.read_text(encoding="utf-8")
        finally:
            make_eval_set.LITE_PATH = real_lite

    check("速览版列出了每一条", all(f"`{it['id']}`" in lite for it in lite_fixture))
    check("速览版带上了分类体系", "产品发布" in lite and "模型与技术" in lite)
    check("速览版带上了重要性标准", "会改变我未来三个月的做法" in lite)
    check("每一条都有开头摘录", lite.count("> ") >= len(lite_fixture),
          f"实际 {lite.count('> ')} 处引用")
    check("摘录被截断，不会退化成全文版", "……" in lite)
    check("速览版的顺序和抽样顺序一致",
          [lite.index(f"`{it['id']}`") for it in lite_fixture]
          == sorted(lite.index(f"`{it['id']}`") for it in lite_fixture),
          "顺序乱了的话，人得来回找对应的行")
    check("速览版确实比全文版短得多", len(lite) < 6000, f"实际 {len(lite)} 字符")

    # 标注向导：它要替用户省掉的，是"在两个文件之间对 20 个十六进制编号"这种麻烦。
    # 所以这里守的是【存盘格式】—— 向导写出来的东西必须能被 compile 正确读回，
    # 否则用户辛苦点完 20 条，最后编译报错，那就彻底白干了。
    with tempfile.TemporaryDirectory() as tmp:
        real_labels = make_eval_set.LABELS_PATH
        make_eval_set.LABELS_PATH = Path(tmp) / "labels.txt"
        make_eval_set.LABELS_PATH.write_text(
            "# 标注表 · 4 条\n# 这是表头，不能被向导弄丢\n\n"
            "w1 |  | \nw2 |  | \nw3 |  | \nw4 |  | \n",
            encoding="utf-8",
        )
        try:
            make_eval_set._rewrite_labels(
                ["w1", "w2", "w3", "w4"],
                {
                    "w1": ("模型与技术", 4, False),
                    "w2": (None, None, True),      # 用户选了"拿不准"
                    "w3": ("产品发布", 2, False),
                    # w4 没答
                },
            )
            written = make_eval_set.LABELS_PATH.read_text(encoding="utf-8")
            back = make_eval_set.parse_labels(make_eval_set.LABELS_PATH)
        finally:
            make_eval_set.LABELS_PATH = real_labels

    check("向导写出的表头被保留", "这是表头，不能被向导弄丢" in written,
          "表头丢了，用户就看不到填法说明")
    check("向导写出的顺序和原顺序一致",
          [written.index(f"w{i} |") for i in (1, 2, 3, 4)] == sorted(written.index(f"w{i} |") for i in (1, 2, 3, 4)))
    check("向导把分类和重要性写成了可编译的格式",
          back.get("w1") == ("模型与技术", 4, False), f"实际 {back.get('w1')}")
    check("「拿不准」被写成问号并保留",
          back.get("w2") == (None, None, True), f"实际 {back.get('w2')}")
    check("没答的条目保持空白，不会被误写成已填",
          back.get("w4") == (None, None, False), f"实际 {back.get('w4')}")
    check("向导写出的文件能被 parse_labels 完整读回（编译的前提）",
          len(back) == 4, f"实际读回 {len(back)} 条")

    # ---------------------------------------------------------- 标注向导的按键
    print("\n[⓪ 评估层 · 标注向导的按键真的存在吗]")

    # 界面上一开始就写着"想看全文就按 m"。第一版处理输入的函数根本不认识 m ——
    # 用户按下去只会得到"看不懂 'm'"，反复试几次，然后觉得这东西不好使。
    # **界面承诺的每一个键，都必须真的存在。**
    #
    # 用脚本化输入来验：把 input() 换成按顺序吐固定答案的假函数。
    import builtins
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cand_path = tmp_dir / "candidates.jsonl"
        lab_path = tmp_dir / "labels.txt"
        marker = "【这是只有全文里才有的标记】"
        cand_path.write_text(
            json.dumps(
                {
                    "id": "w1",
                    "title": "第 1 条",
                    "source": "测试源",
                    "body": "正文内容。" * 60 + marker,   # 开头 400 字里看不到这个标记
                    "gold": {"category": None, "importance": None},
                },
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        lab_path.write_text("# 表头\n\nw1 |  | \n", encoding="utf-8")

        real_cand = make_eval_set.CANDIDATES_PATH
        real_lab = make_eval_set.LABELS_PATH
        make_eval_set.CANDIDATES_PATH = cand_path
        make_eval_set.LABELS_PATH = lab_path

        # 依次输入：看全文 → 选分类 2 → 重要性输入 0（应被拒绝）→ 输入 4 → 退出
        scripted = iter(["m", "2", "0", "4", "q"])
        real_input = builtins.input

        def fake_input(prompt: str = "") -> str:
            try:
                return next(scripted)
            except StopIteration:
                raise EOFError

        builtins.input = fake_input
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                make_eval_set.cmd_wizard(None)
            wizard_out = buffer.getvalue()
            saved_labels = lab_path.read_text(encoding="utf-8")
        finally:
            builtins.input = real_input
            make_eval_set.CANDIDATES_PATH = real_cand
            make_eval_set.LABELS_PATH = real_lab

    check("按 m 真的能看全文（界面承诺的键必须存在）",
          marker in wizard_out,
          "按了 m 却看不到全文 —— 用户会以为自己按错了")
    check("按 m 不会被当成无效输入",
          "'m'" not in wizard_out, "把 m 当成看不懂的输入回给用户了")
    check("重要性不接受「拿不准」（0）",
          "'0'" in wizard_out,
          "如果 0 被当成拿不准，用户刚选好的分类会在存盘时一起丢掉")
    check("向导最终把答案正确存了下来",
          "w1 | 模型与技术 | 4" in saved_labels,
          f"实际存成：{[l for l in saved_labels.splitlines() if l.startswith('w1')]}")

    # ---------------------------------------------------------- ④ 分发层
    print("\n[④ 分发层 · 日报渲染与审核解析]")

    from src.archive import parse_review, render_digest, render_review

    def fake_record(item_id, category, importance, title):
        return {
            "id": item_id,
            "title": title,
            "source": "测试源",
            "status": "ok",
            "output": {
                "title": title,
                "summary": f"{title} 的摘要。",
                "category": category,
                "importance": importance,
                "entities": ["实体甲"],
                "so_what": "影响。",
                "evidence": ["原文一句"],
            },
            "problems": [],
        }

    digest_records = [
        fake_record("d1", "模型与技术", 4, "条目一"),
        fake_record("d2", "模型与技术", 3, "条目二"),
        fake_record("d3", "其他", 5, "条目三"),
        fake_record("d4", "我瞎编的分类", 2, "条目四"),  # 不在配置里 -> 归入"其他"
    ]
    digest = render_digest(digest_records, {"provider": "p", "model": "m"}, stamp="test", task_cfg=task_cfg)

    for record in digest_records:
        occurrences = digest.count(f"`{record['id']}`")
        check(f"日报里 {record['id']} 只出现一次", occurrences == 1,
              f"实际出现 {occurrences} 次（分组顺序没去重时会渲染两遍）")

    # 日报是【先按分类分组，组内再按重要性降序】——
    # 不是全局按重要性排。分组是为了好读，组内排序是为了先看到重要的。
    # 我第一版测试断言写成了全局排序，所以这里把两件事分别断。
    def section(text: str, heading: str) -> str:
        """取出一级标题下的内容。

        必须用 "\n## " 锚定，不能用 "## " ——
        后者会匹配到 "### " 里面（第三个字符起正好是 "## "），
        于是切出来的片段是空的，而报错信息只会说"找不到子串"，
        完全看不出真正的原因。
        """
        marker = f"\n## {heading}\n"
        if marker not in text:
            return ""
        return text.split(marker, 1)[1].split("\n## ", 1)[0]

    tech_group = section(digest, "模型与技术")
    other_group = section(digest, "其他")
    check("日报里确实有模型与技术分组", "条目一" in tech_group, f"片段={tech_group[:60]!r}")
    check("同一个分类内按重要性降序（4 排在 3 前面）",
          tech_group.index("条目一") < tech_group.index("条目二"))
    check("兜底分组内也按重要性降序（5 排在 2 前面）",
          other_group.index("条目三") < other_group.index("条目四"))
    check("分类之间按 config/task.json 的顺序，不是按重要性打乱",
          digest.index("## 模型与技术") < digest.index("## 其他"))
    check("分类不在允许列表里的条目被归入「其他」", "`d4`" in digest)
    check("未审核的日报带明显标记", "未审核" in digest)
    check("未审核的日报给出下一步命令", "review --apply test" in digest)

    reviewed = {
        "d1": {"included": True, "category": "产品发布", "importance": 5, "ambiguous": False},
        "d2": {"included": False, "category": None, "importance": None, "ambiguous": False},
        "d3": {"included": True, "category": None, "importance": 4, "ambiguous": False},
        "d4": {"included": True, "category": None, "importance": None, "ambiguous": False},
    }
    final = render_digest(
        digest_records, {"provider": "p", "model": "m"}, stamp="test",
        task_cfg=task_cfg, reviewed=reviewed, final=True,
    )
    check("被标为不收录的条目从终版里消失", "`d2`" not in final)
    check("终版使用人工改判后的重要性（d3 从 5 改成 4）",
          final.index("`d1`") < final.index("`d3`"), "d1 改判为 5，应排在 d3(4) 前面")
    check("终版标出被人工修正的条目", "人工修正" in final)
    check("终版标记为已审核", "已审核" in final)

    review_text = render_review(digest_records, {}, stamp="test", task_cfg=task_cfg)
    with tempfile.TemporaryDirectory() as tmp:
        rp = Path(tmp) / "review.txt"
        rp.write_text(review_text, encoding="utf-8")
        parsed = parse_review(rp)
        check("审核表里的条目都能被读回来", len(parsed) == 4, f"实际 {len(parsed)}")
        check("全部留空 = 全部通过",
              all(d["included"] for d in parsed.values()))
        # 分类和重要性是【预填模型的判断】，不是空的。
        # 空的话人就得从头填一遍，审核成本会高到没人做第二次 ——
        # 而审核一旦没人做，回流评估集这条闭环就断了。
        check("分类列预填了模型的判断（不改 = 认可）",
              all(d["category"] is not None for d in parsed.values()),
              f"实际 {[d['category'] for d in parsed.values()]}")
        check("重要性列也预填了模型的判断",
              all(d["importance"] is not None for d in parsed.values()))
        check("说明文字与列的实际情况一致（收录列才是留空即通过）",
              "收录列留空就是通过" in review_text,
              "说明说'都留空'但两列是预填的，人读了会不知道该怎么做")
        check("审核表把模型的分类原样带上（人只需改不对的）",
              any("模型与技术" in line for line in review_text.splitlines()))

        edited = review_text.replace("d1 |  | 模型与技术 | 4", "d1 | x | 观点与分析 | 5")
        rp.write_text(edited, encoding="utf-8")
        parsed2 = parse_review(rp)
        check("x 被识别为不收录", parsed2["d1"]["included"] is False)
        check("改写的分类被读出来", parsed2["d1"]["category"] == "观点与分析",
              f"实际 {parsed2['d1']['category']}")
        check("改写的重要性被读出来", parsed2["d1"]["importance"] == 5)

        rp.write_text(review_text.replace("d1 |  | 模型与技术 | 4", "d1 |  | ? | "), encoding="utf-8")
        parsed3 = parse_review(rp)
        check("问号被识别为存疑样本", parsed3["d1"]["ambiguous"] is True)

    # 回流评估集：这是整套系统里唯一一处"它会自己变强"的地方，
    # 所以它必须幂等 —— 手滑跑两次不该让评估集里多出一倍的重复样本。
    from src import archive as archive_mod

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "eval_set.jsonl"
        target.write_text(
            '{"id": "x1", "gold": {"category": "其他", "importance": 3}}\n', encoding="utf-8"
        )
        original_path = archive_mod.EVAL_SET_PATH
        archive_mod.EVAL_SET_PATH = target
        try:
            fresh = [
                {"id": "x1", "body": "已在评估集里", "gold": {"category": "其他", "importance": 3}},
                {"id": "x2", "body": "新的", "gold": {"category": "产品发布", "importance": 4}},
                {"id": "x3", "body": "也是新的", "gold": {"category": "政策监管", "importance": 5}},
            ]
            added = archive_mod.append_to_eval_set(fresh)
            check("回流时已存在的 id 被跳过", added == 2, f"实际追加 {added} 条")
            check("再次回流不会重复追加（幂等）",
                  archive_mod.append_to_eval_set(fresh) == 0, "手滑跑两次会让评估集翻倍")
            ids = archive_mod.load_eval_set_ids()
            check("回流后评估集内容正确", ids == {"x1", "x2", "x3"}, f"实际 {ids}")
        finally:
            archive_mod.EVAL_SET_PATH = original_path

    # ---------------------------------------------------------- ② 厂商配置
    print("\n[② 模型层 · 厂商配置与模型选择]")

    from src.config import load_providers
    from probe import suggest_model

    providers_cfg = load_providers().get("providers", {})
    check("配置里有多家厂商可选", len(providers_cfg) >= 3, f"实际 {len(providers_cfg)} 家")

    # status 面板靠 console_url 打印"去哪申请密钥"。
    # 少了它，面板只会说一句"去申请一个密钥" —— 对没做过这件事的人，那等于没说。
    missing_url = [
        name for name, cfg in providers_cfg.items()
        if not str(cfg.get("console_url", "")).startswith("http")
    ]
    check("每家厂商都配了申请密钥的网址", not missing_url, f"缺网址：{missing_url}")

    missing_env = [name for name, cfg in providers_cfg.items() if not cfg.get("api_key_env")]
    check("每家厂商都配了环境变量名", not missing_env, f"缺：{missing_env}")

    missing_base = [
        name for name, cfg in providers_cfg.items()
        if not str(cfg.get("base_url", "")).startswith("http")
    ]
    check("每家厂商都配了接口地址", not missing_base, f"缺：{missing_base}")

    check("被推荐的那家（zhipu）写了注册提示",
          bool(providers_cfg.get("zhipu", {}).get("signup_hint")),
          "推荐哪家、怎么注册，是用户第一个会问的问题")

    # 模型选择：这里守的是一条会让用户第一次就卡死的路径。
    # 有的厂商不提供 /models 接口 —— 那和"密钥好不好"毫无关系，
    # 绝不能因此告诉用户"无法测试"。
    check("列不出模型名时，退回用配置里写的那个去试",
          suggest_model([], "glm-x") == ("glm-x", True),
          "第一版这里返回 None，于是 probe 报「无法测试」——把好好的密钥判成了死局")
    check("既列不出模型、配置里也没写，才真的没法测",
          suggest_model([], None) == (None, False))
    check("配置里写的模型在列表里时优先用它",
          suggest_model(["a-flash", "b"], "b") == ("b", True))
    check("配置里没有的模型名时，自己挑一个",
          suggest_model(["big-expensive", "small-flash"], None) == ("small-flash", False),
          "没有线索时优先挑便宜的 —— 基线和试跑都不需要最贵的模型")
    check("挑模型时会排除明显不是聊天的（embedding 等）",
          suggest_model(["text-embedding-3", "chat-mini"], None) == ("chat-mini", False))
    check("全都不是聊天模型时也不会挑出空",
          suggest_model(["text-embedding-3"], None) == ("text-embedding-3", False))

    # ---- 编码宽容度：这条是为中文 Windows 用户守的
    #
    # 记事本默认把文件存成 GBK，而 labels.txt 和 eval_set.jsonl
    # 恰恰是明确希望用户【手工编辑】的两个文件。
    # 用 utf-8 硬读，用户会得到一个 UnicodeDecodeError 的英文报错栈，
    # 完全看不出"这是编码问题"。
    #
    # 这个 bug 是被自己的测试意外挖出来的：测试数据用 PowerShell 的
    # Add-Content 追加（它按 GBK 写），下一次 compile 就直接抛栈了。
    from src.config import read_text_smart

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        utf8_path = tmp_dir / "u8.txt"
        utf8_path.write_text("中文内容：模型与技术\n", encoding="utf-8")
        check("能读 UTF-8 文件", "模型与技术" in read_text_smart(utf8_path))

        bom_path = tmp_dir / "bom.txt"
        bom_path.write_text("中文内容\n", encoding="utf-8-sig")
        check("能读带 BOM 的 UTF-8（记事本另存为 UTF-8 会带 BOM）",
              read_text_smart(bom_path).strip() == "中文内容",
              f"实际 {read_text_smart(bom_path)!r}")

        gbk_path = tmp_dir / "gbk.txt"
        gbk_path.write_bytes("中文内容：模型与技术\n".encode("gbk"))
        check("能读 GBK（中文 Windows 记事本的默认编码）",
              "模型与技术" in read_text_smart(gbk_path),
              "旧代码在这里抛 UnicodeDecodeError，用户看不懂")

        broken_path = tmp_dir / "broken.bin"
        broken_path.write_bytes(b"\xff\xfe\x00\x01\x02")
        try:
            read_text_smart(broken_path)
            check("遇到彻底读不了的文件也不会崩", True)
        except Exception as exc:  # noqa: BLE001
            check("遇到彻底读不了的文件也不会崩", False, f"抛了 {type(exc).__name__}")

    # 审核表也是用户手写的 —— out/review/<时间戳>.txt
    from src.archive import parse_review as _parse_review

    with tempfile.TemporaryDirectory() as tmp:
        review_gbk = Path(tmp) / "review.txt"
        review_gbk.write_bytes(
            "```text\nabc123 |  | 模型与技术 | 4\n```\n".encode("gbk")
        )
        parsed_gbk = _parse_review(review_gbk)
        entry = parsed_gbk.get("abc123", {})
        check("审核表是 GBK 编码时也能读",
              entry.get("category") == "模型与技术" and entry.get("importance") == 4,
              f"实际 {entry}")

    # 往用户手改过的文件里追加内容：不能制造出"两种编码混在一起"的文件。
    # 混合编码的文件之后无论用哪种编码读，总有一半是乱码，
    # 而且很难看出是从哪一行开始坏的。
    from src import archive as archive_mod2

    with tempfile.TemporaryDirectory() as tmp:
        mixed = Path(tmp) / "eval_set.jsonl"
        mixed.write_bytes(
            '{"id":"g1","gold":{"category":"模型与技术","importance":4}}\n'.encode("gbk")
        )
        real_eval_path = archive_mod2.EVAL_SET_PATH
        archive_mod2.EVAL_SET_PATH = mixed
        try:
            added = archive_mod2.append_to_eval_set(
                [{"id": "g2", "body": "x", "gold": {"category": "其他", "importance": 3}}]
            )
            ids_after = archive_mod2.load_eval_set_ids()
            raw_after = mixed.read_bytes()
        finally:
            archive_mod2.EVAL_SET_PATH = real_eval_path

        check("往评估集追加时能读已有的 GBK 内容", added == 1, f"实际追加 {added} 条")
        check("追加没有丢掉原有的行", ids_after == {"g1", "g2"}, f"实际 {ids_after}")
        try:
            raw_after.decode("utf-8")
            is_utf8 = True
        except UnicodeDecodeError:
            is_utf8 = False
        check("追加后整个文件被统一成 UTF-8（不会变成混合编码）", is_utf8,
              "GBK 的旧内容 + UTF-8 的新行 = 之后怎么读都是乱码")

    # ---------------------------------------------------------- ⑤ 权限层
    print("\n[⑤ 权限层 · 每日运行与健康记录]")

    from src import runner as runner_mod
    from src.runner import Health, StepSkipped, run_daily

    with tempfile.TemporaryDirectory() as tmp:
        alert_path = Path(tmp) / "ALERT.md"
        original_alert = runner_mod.ALERT_PATH
        runner_mod.ALERT_PATH = alert_path
        try:
            health = Health(Path(tmp) / "health.json", escalate_after=3, keep_runs=3)
            calls: list[str] = []

            def ok_step(name):
                def fn():
                    calls.append(name)
                    return f"{name} 完成"
                return fn

            def fail_step(name):
                def fn():
                    calls.append(name)
                    raise RuntimeError(f"{name} 炸了")
                return fn

            def skip_step(name):
                def fn():
                    calls.append(name)
                    raise StepSkipped("没有新内容")
                return fn

            # ---- 正常一轮
            calls.clear()
            out = run_daily([("a", ok_step("a")), ("b", ok_step("b"))], health=health, echo=False)
            check("全成功时运行标记为成功", out["ok"] is True)
            check("每一步都写进了健康表",
                  health.get("a")["last_summary"] == "a 完成"
                  and health.get("b")["last_summary"] == "b 完成")
            check("成功时连续失败计数为 0", health.get("a")["consecutive_failures"] == 0)

            # ---- 默认遇错即停
            calls.clear()
            out = run_daily(
                [("a", ok_step("a")), ("b", fail_step("b")), ("c", ok_step("c"))],
                health=health, echo=False,
            )
            check("默认遇错即停，不执行后面的步骤", calls == ["a", "b"], f"实际执行了 {calls}")
            check("停在失败的那一步", out["stopped_at"] == "b")
            check("运行被标记为失败", out["ok"] is False)
            check("失败计数 +1", health.get("b")["consecutive_failures"] == 1)
            check("失败原因被记下来", "炸了" in (health.get("b")["last_error"] or ""))

            # ---- continue-on-error
            calls.clear()
            out = run_daily(
                [("a", ok_step("a")), ("b", fail_step("b")), ("c", ok_step("c"))],
                health=health, continue_on_error=True, echo=False,
            )
            check("--continue-on-error 时会继续执行", calls == ["a", "b", "c"], f"实际 {calls}")
            check("继续执行也仍然标记本次运行失败", out["ok"] is False)
            check("失败计数累加到 2", health.get("b")["consecutive_failures"] == 2,
                  f"实际 {health.get('b')['consecutive_failures']}")
            check("未达阈值时不该有警报文件", not alert_path.exists())

            # ---- 达到阈值 -> 升级
            run_daily([("b", fail_step("b"))], health=health, echo=False)
            check("连续失败达阈值触发警报", health.get("b")["consecutive_failures"] == 3,
                  f"实际 {health.get('b')['consecutive_failures']}")
            check("警报文件被写出来", alert_path.exists())
            check("警报里写明了是哪个任务", "`b`" in alert_path.read_text(encoding="utf-8"))
            check("警报里说明了怎么恢复",
                  "run.cmd daily" in alert_path.read_text(encoding="utf-8"))

            # ---- 恢复 -> 自动撤销警报
            out = run_daily([("b", ok_step("b"))], health=health, echo=False)
            check("恢复后失败计数清零", health.get("b")["consecutive_failures"] == 0)
            check("恢复后警报文件被自动删除", not alert_path.exists(), "长期悬挂的警报等于没有警报")
            check("撤销动作被记录在本次结果里", out.get("alert_cleared") is True)

            # ---- 跳过不是失败：这一层最容易做错的地方
            health2 = Health(Path(tmp) / "health2.json", escalate_after=3, keep_runs=5)
            for _ in range(3):
                run_daily([("x", fail_step("x"))], health=health2, echo=False)
            check("（前置）x 已经连续失败 3 次", health2.get("x")["consecutive_failures"] == 3)
            check("（前置）警报已发出", alert_path.exists())

            out = run_daily([("x", skip_step("x")), ("y", ok_step("y"))], health=health2, echo=False)
            check("跳过时本次运行仍然是成功的", out["ok"] is True,
                  "把跳过一次算作失败，连续三天没新闻就会触发假警报")
            check("跳过会把连续失败计数清零", health2.get("x")["consecutive_failures"] == 0,
                  "不清零就会得到一个长期悬挂、永远不会撤销的警报")
            check("跳过被单独统计", out["skipped"] == 1, f"实际 {out['skipped']}")
            check("跳过的摘要写明了原因",
                  "没有新内容" in str(health2.get("x")["last_summary"]))
            check("跳过之后警报被撤销", not alert_path.exists())

            # ---- 运行历史有上限
            health3 = Health(Path(tmp) / "health3.json", escalate_after=3, keep_runs=3)
            for _ in range(6):
                run_daily([("z", ok_step("z"))], health=health3, echo=False)
            check("运行历史被裁剪到上限", len(health3.data["runs"]) == 3,
                  f"实际 {len(health3.data['runs'])} 条（健康表是'最近怎么样'，不是永久档案）")

            # ---- 健康表能读回来
            # 注意这里是 a/b/c 三个：修好 --continue-on-error 之前，
            # c 根本不会被执行，所以这个断言当时"通过"了 ——
            # 它是被 bug 的行为校准出来的。修好之后它才开始失败。
            # 这就是为什么改完一个 bug 必须把整套测试重跑一遍，
            # 而不是只看你刚修的那一条。
            table = health.table()
            check("健康表列出了所有执行过的任务", {row["task"] for row in table} == {"a", "b", "c"},
                  f"实际 {[row['task'] for row in table]}")
        finally:
            runner_mod.ALERT_PATH = original_alert

    # "还没配密钥"不是"坏了"。
    #
    # 这条漏网过一次：daily 里的 pipeline 步骤把"没有密钥"记成了一次失败。
    # 后果很具体 —— 你随手试跑三次 run.cmd daily，就会收到一个 out/ALERT.md，
    # 上面写着"pipeline 连续失败 3 次"，而你根本没做错任何事。
    #
    # 假警报的代价不是打扰你一次，是你会开始无视所有警报。
    print("\n[⑤ 权限层 · 还没配密钥不算失败]")

    import src.llm as _llm_module

    class _NoKeyClient:
        def __init__(self, *args, **kwargs):
            from src.secrets import MissingKey

            raise MissingKey("测试用：没有密钥")

    _real_client_class = _llm_module.LLMClient
    _llm_module.LLMClient = _NoKeyClient
    try:
        runner_mod._step_pipeline(stub=False)
        check("没配密钥时记为「跳过」而不是「失败」", False,
              "它没有抛异常，说明走上了一条不该走的路")
    except StepSkipped as exc:
        check("没配密钥时记为「跳过」而不是「失败」", "密钥" in str(exc),
              f"抛的是 StepSkipped，但信息里没提密钥：{exc}")
    except Exception as exc:  # noqa: BLE001
        check("没配密钥时记为「跳过」而不是「失败」", False,
              f"抛了 {type(exc).__name__} —— 会被记成一次失败，试跑三次就误报警")
    finally:
        _llm_module.LLMClient = _real_client_class

    # ---------------------------------------------------------- 文档不写死数字
    print("\n[⑤ 文档 · 不写会过期的数字]")

    # 这条守的是一类很隐蔽的假警报：
    #
    # 文档里原来写着「你应该看到 129 / 129」。而随着项目长大，
    # 实际输出变成了 186 / 186 —— 用户照着文档一对，发现对不上，
    # 会以为是自己弄坏了什么，然后跑来问一个根本不存在的问题。
    #
    # **任何写死在文档里的、会随项目变化的数字，都会变成假警报。**
    # 所以干脆让测试盯着：文档里不许出现具体项数。
    import re as _re_docs

    stale: list[str] = []
    doc_files = sorted((PROJECT_ROOT / "docs").glob("*.md")) + [PROJECT_ROOT / "README.md"]
    for doc in doc_files:
        if not doc.exists():
            continue
        for lineno, line in enumerate(read_text_smart(doc).splitlines(), 1):
            if _re_docs.search(r"全部通过\s+\d+\s*/", line):
                stale.append(f"{doc.name}:{lineno}（写死了「全部通过 x / y」）")
            if _re_docs.search(r"#\s*\d+\s*项", line):
                stale.append(f"{doc.name}:{lineno}（命令注释里写死了项数）")

    check("文档里没有写死的自检项数", not stale,
          "这些地方会随项目长大而变成假警报：" + "；".join(stale))

    # README 的目录树是一份**承诺**：它告诉用户"这里有什么"。
    # 里面的文件名必须真的存在，而且实际的模块必须都在里面 ——
    # 一个漏了 config.py 的地图，会让读它的人以为那部分不存在。
    #
    # 注意正则里的后缀顺序：长的必须排在前面。
    # 我第一版写成 (py|md|json|jsonl)，于是 json 先匹配掉了 jsonl 的前四个字符，
    # 检查报出一个【并不存在】的缺失文件 —— 一个误报的检查，
    # 和没有检查一样糟，甚至更糟。
    readme_text = read_text_smart(PROJECT_ROOT / "README.md")
    tree_files = set(
        _re_docs.findall(
            r"[\u251c\u2514]\u2500\s*([\w\-.]+\.(?:jsonl|json|py|md|cmd|txt))", readme_text
        )
    )
    check("README 目录树解析出了合理的文件数", len(tree_files) >= 20,
          f"只解析出 {len(tree_files)} 个，正则该改")

    tree_missing = sorted(n for n in tree_files if not list(PROJECT_ROOT.rglob(n)))
    check("README 目录树里提到的文件都真的存在", not tree_missing,
          f"树上写了但磁盘上没有：{tree_missing}")

    real_modules = {
        p.name
        for p in list((PROJECT_ROOT / "src").glob("*.py")) + list((PROJECT_ROOT / "tools").glob("*.py"))
        if p.name != "__init__.py"
    }
    not_in_tree = sorted(real_modules - tree_files)
    check("实际存在的模块都写进了 README 目录树", not not_in_tree,
          f"漏了：{not_in_tree}")

    # ---------------------------------------------------------- 入口脚本
    print("\n[⑤ 入口脚本 · 编码、换行、引用]")

    import re as _re

    # 这几条都是真的踩过的坑，不是假想的：
    #   · cmd.exe 用 OEM 码页读 .cmd，非 ASCII 内容会被撕碎，
    #     甚至把注释行解析成命令（"xxx 不是内部或外部命令"）
    #   · LF-only 的批处理文件里 goto 标签会失效
    #     （"The system cannot find the batch label specified"）
    #   · 脚本改名之后，启动器里写死的路径会静默失效 ——
    #     双击没反应，而且不报任何错
    for launcher in ("run.cmd", "开始标注.cmd", "看下一步.cmd"):
        path = PROJECT_ROOT / launcher
        if not path.exists():
            check(f"{launcher} 存在", False, "文件没了")
            continue
        raw = path.read_bytes()
        non_ascii = sum(1 for b in raw if b >= 128)
        check(f"{launcher} 是纯 ASCII", non_ascii == 0,
              f"有 {non_ascii} 个非 ASCII 字节，cmd.exe 会按 OEM 码页撕碎它")
        bare_lf = raw.replace(b"\r\n", b"").count(b"\n")
        check(f"{launcher} 全部用 CRLF 换行", bare_lf == 0,
              f"有 {bare_lf} 个裸 LF —— 批处理里的 goto 标签会找不到")

        text = raw.decode("ascii", errors="replace")
        referenced = _re.findall(r"(?:tools|src)\\[\w.]+\.py", text)
        missing = [r for r in referenced if not (PROJECT_ROOT / r.replace("\\", "/")).exists()]
        check(f"{launcher} 里引用的脚本都存在", not missing, f"缺：{missing}")
        check(f"{launcher} 在切换编码后才调用 Python",
              text.index("chcp 65001") < text.index("python "),
              "先调 Python 再切编码，中文输出会是乱码")

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
