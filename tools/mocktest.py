"""全链路自检：用一个假服务端，验证真客户端。

【它和 selftest 的分工】

    selftest   纯逻辑。解析、校验、预算、脱敏。不碰网络，毫秒级。
    mocktest   全链路。真的发 HTTP、真的收响应、真的重试、真的写账本。
               服务端是本地起的一个假 OpenAI，不连外网、不花钱。

【为什么必须有这个文件】

llm.py 里的 provider 适配器是整套系统里【风险最大又最难验证】的一块。
它出错的方式都很贵：
    - 把 usage 字段读错  -> 成本统计全错，你却以为很省钱
    - 重试逻辑写反了      -> 该重试的不重试，丢数据
    - 把 4xx 当可重试     -> 密钥错了却重试三次，白等
    - 异常信息里带出密钥  -> 密钥泄漏到日志

而这些在真实 API 上验证，每一次都要花钱、要网络、要等。
更麻烦的是：真实服务端【不会按你的要求失败】——
你想测"429 之后能不能退避重试"，得先真的把配额打满。

假服务端能按命令失败。这是它的全部价值：

    **能被人为触发的失败，才是能被测试的失败。**

用法：
    python tools/mocktest.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.llm as llm_module  # noqa: E402
from src.ledger import Budget, BudgetExceeded, Ledger  # noqa: E402
from src.llm import FatalLLMError, LLMClient, RetryableLLMError  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []

MOCK_KEY = "sk-mock-abcdefghijklmnop-9876"


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(name)
        print(f"  [OK] {name}")
    else:
        FAILED.append(name)
        print(f"  [!!] {name}  {detail}")


def expect_raises(name: str, exc_type: type, fn, contains: str = "") -> object | None:
    try:
        fn()
    except exc_type as exc:
        if contains and contains not in str(exc):
            check(name, False, f"异常信息里没有 {contains!r}")
        else:
            check(name, True)
        return exc
    except Exception as exc:  # noqa: BLE001
        check(name, False, f"抛的是 {type(exc).__name__}，期望 {exc_type.__name__}：{exc}")
        return exc
    check(name, False, "没有抛出异常")
    return None


# ------------------------------------------------------------------ 假服务端


def _normal_response(model: str) -> dict:
    return {
        "id": "chatcmpl-mock-1",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "title": "模拟标题",
                            "summary": "模拟摘要内容。",
                            "category": "模型与技术",
                            "importance": 4,
                            "entities": ["甲公司"],
                            "so_what": "模拟影响。",
                            "evidence": ["模拟原文片段"],
                        },
                        ensure_ascii=False,
                    ),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }


class MockHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "MockOpenAI/0.1"

    def log_message(self, *args) -> None:  # 静音，否则测试输出会被日志淹没
        return

    # ---------------- 工具

    def _send(self, code: int, payload, *, headers: dict | None = None) -> None:
        body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _prompt_text(self, body: dict) -> str:
        parts = []
        for message in body.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(str(b.get("text", "")) for b in content if isinstance(b, dict))
        return "\n".join(parts)

    def _bump(self, key: str) -> int:
        """给某个触发器计数。返回这是第几次。"""
        state = self.server.state  # type: ignore[attr-defined]
        state[key] = int(state.get(key, 0)) + 1
        return state[key]

    # ---------------- 路由

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/").endswith("/models"):
            self._send(
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "mock-model-1", "object": "model"},
                        {"id": "mock-embedding-1", "object": "model"},
                        {"id": "mock-model-flash", "object": "model"},
                    ],
                },
            )
            return
        self._send(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, {"error": {"message": "not found"}})
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send(400, {"error": {"message": "bad request body"}})
            return

        self.server.requests.append(  # type: ignore[attr-defined]
            {
                "path": self.path,
                "body": body,
                "authorization": self.headers.get("Authorization", ""),
            }
        )

        text = self._prompt_text(body)
        model = body.get("model", "unknown")

        # ---- 触发失败：真实服务端不会按你要求失败，假服务端会

        if "TRIGGER_FATAL_401" in text:
            # 故意在错误信息里回显密钥，用来验证客户端会不会把它泄漏到日志里
            self._send(401, {"error": {"message": f"invalid api key: {MOCK_KEY}"}})
            return

        if "TRIGGER_FATAL_404" in text:
            self._send(404, {"error": {"message": "model not found"}})
            return

        if "TRIGGER_RETRY_500_ALWAYS" in text:
            self._send(500, {"error": {"message": "internal server error"}})
            return

        if "TRIGGER_RETRY_500_ONCE" in text:
            if self._bump("retry500") == 1:
                self._send(500, {"error": {"message": "temporary failure"}})
                return
            self._send(200, _normal_response(model))
            return

        if "TRIGGER_RETRY_429" in text:
            if self._bump("retry429") == 1:
                self._send(429, {"error": {"message": "rate limited"}}, headers={"Retry-After": "0"})
                return
            self._send(200, _normal_response(model))
            return

        # ---- 触发畸形返回：真实服务端偶尔也会这样

        if "TRIGGER_NO_CHOICES" in text:
            self._send(200, {"id": "x", "choices": [], "usage": {}})
            return

        if "TRIGGER_BAD_JSON" in text:
            self._send(200, b"this is definitely not json")
            return

        if "TRIGGER_FENCED_JSON" in text:
            payload = _normal_response(model)
            payload["choices"][0]["message"]["content"] = (
                "好的，以下是结果：\n```json\n" + payload["choices"][0]["message"]["content"] + "\n```\n希望有帮助。"
            )
            self._send(200, payload)
            return

        if "TRIGGER_CONTENT_LIST" in text:
            payload = _normal_response(model)
            payload["choices"][0]["message"]["content"] = [
                {"type": "text", "text": "第一块"},
                {"type": "text", "text": "第二块"},
            ]
            self._send(200, payload)
            return

        if "TRIGGER_NO_USAGE" in text:
            payload = _normal_response(model)
            payload.pop("usage", None)
            self._send(200, payload)
            return

        self._send(200, _normal_response(model))


class MockServer:
    def __init__(self) -> None:
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), MockHandler)
        self.httpd.requests = []  # type: ignore[attr-defined]
        self.httpd.state = {}  # type: ignore[attr-defined]
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self) -> "MockServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def requests(self) -> list[dict]:
        return self.httpd.requests  # type: ignore[attr-defined]


class TimeShim:
    """把 sleep 换掉，但保留 perf_counter。

    为什么需要它：真实的重试退避是 1.5 秒起步、指数增长。
    一个测重试的套件如果真等下去，会花掉十几秒 —— 于是你不会经常跑它，
    于是它就不再是"每次改完都跑一遍"的东西了。

    这里不改变被测逻辑，只是把"等待"这件事的代价抹掉，
    同时把每次等待的时长记下来，好断言退避真的发生了。
    """

    def __init__(self, real) -> None:
        self._real = real
        self.slept: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(float(seconds), 3))

    def __getattr__(self, name: str):
        return getattr(self._real, name)


# ------------------------------------------------------------------ 测试


def build_client(server: MockServer, *, json_mode_supported: bool = True, price: bool = True, **kwargs):
    providers = {
        "providers": {
            "mock": {
                "label": "本地假服务端",
                "base_url": server.base_url,
                "api_key_env": "MOCK_API_KEY",
                "default_model": "mock-model-1",
                "model_verified": True,
                "supports_json_mode": json_mode_supported,
                "price_cny_per_1m": {"input": 1.0, "output": 2.0} if price else {"input": None, "output": None},
                "price_verified_on": "test",
            }
        }
    }
    policy = {
        "network": {"allowed_hosts": ["127.0.0.1"], "timeout_seconds": 10, "max_retries": 3},
        "budget": {
            "max_calls_per_run": 50,
            "max_cost_per_run_cny": 100.0,
            "max_cost_per_day_cny": 100.0,
        },
        "ledger": {"store_full_prompt": False},
    }
    ledger = kwargs.pop("ledger", None)
    budget = kwargs.pop("budget", None)
    return (
        LLMClient(
            "mock",
            providers_cfg=providers,
            policy_cfg=policy,
            ledger=ledger,
            budget=budget,
            **kwargs,
        ),
        policy,
    )


def main() -> int:
    os.environ["MOCK_API_KEY"] = MOCK_KEY

    print()
    print("=" * 68)
    print("  全链路自检（本地假服务端，不连外网，不花钱）")
    print("=" * 68)

    import tempfile

    real_time = llm_module.time
    shim = TimeShim(real_time)
    llm_module.time = shim  # type: ignore[assignment]

    try:
        with MockServer() as server, tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger({}, path=Path(tmp) / "ledger.jsonl")
            budget = Budget(
                {
                    "budget": {
                        "max_calls_per_run": 50,
                        "max_cost_per_run_cny": 100.0,
                        "max_cost_per_day_cny": 100.0,
                    }
                },
                ledger=ledger,
            )
            client, policy = build_client(server, ledger=ledger, budget=budget)

            # ---------------------------------------------------- 正常路径
            print("\n[② 模型层 · 正常调用]")

            result = client.chat(
                [{"role": "user", "content": "给我一条正常结果"}],
                temperature=0.2,
                max_tokens=500,
                json_mode=True,
                purpose="mocktest",
            )
            check("能拿到文本", "模拟摘要内容" in result.text, f"实际 {result.text[:60]!r}")
            check("provider 名正确", result.provider == "mock", f"实际 {result.provider}")
            check("model 名正确", result.model == "mock-model-1", f"实际 {result.model}")
            check("prompt_tokens 被正确读出", result.prompt_tokens == 100, f"实际 {result.prompt_tokens}")
            check("completion_tokens 被正确读出", result.completion_tokens == 50, f"实际 {result.completion_tokens}")
            check("total_tokens 被正确读出", result.total_tokens == 150, f"实际 {result.total_tokens}")
            check("延迟被记录", result.latency_ms >= 0, f"实际 {result.latency_ms}")
            check("request_id 被取出", result.request_id == "chatcmpl-mock-1", f"实际 {result.request_id}")
            check(
                "成本按单价正确计算（100 tok × 1元/1M + 50 tok × 2元/1M = 0.0002）",
                abs(result.cost_cny - 0.0002) < 1e-9,
                f"实际 {result.cost_cny}",
            )
            check("成本已知时标记为已知", result.cost_known is True)
            check("一次成功时 attempts 为 1", result.attempts == 1, f"实际 {result.attempts}")

            # 请求侧：真的把该带的东西带上了
            sent = server.requests[-1]
            check("请求打到 /chat/completions", sent["path"].endswith("/chat/completions"), sent["path"])
            check("带上 Bearer 认证头", sent["authorization"] == f"Bearer {MOCK_KEY}")
            check("payload 带 model", sent["body"].get("model") == "mock-model-1")
            check("payload 带 temperature", sent["body"].get("temperature") == 0.2)
            check("payload 带 max_tokens", sent["body"].get("max_tokens") == 500)
            check(
                "支持 JSON 模式时带上 response_format",
                sent["body"].get("response_format") == {"type": "json_object"},
                f"实际 {sent['body'].get('response_format')}",
            )

            # ---------------------------------------------------- 账本
            print("\n[⓪ 账本]")
            rows = ledger.read_today()
            check("每次调用写进账本", len(rows) == 1, f"实际 {len(rows)} 行")
            check("账本记录了 purpose", rows[0].get("purpose") == "mocktest", f"实际 {rows[0].get('purpose')}")
            check("账本记录了成本", abs(float(rows[0].get("cost_cny") or 0) - 0.0002) < 1e-9)
            check("账本记录了延迟", "latency_ms" in rows[0])
            check("账本里没有密钥明文", MOCK_KEY not in ledger.path.read_text(encoding="utf-8"))

            # ---------------------------------------------------- 单价缺失
            print("\n[② 成本统计 · 单价未配置时]")
            no_price_client, _ = build_client(server, price=False, ledger=ledger, budget=budget)
            no_price_result = no_price_client.chat([{"role": "user", "content": "单价缺失"}])
            check("单价缺失时不假装知道成本", no_price_result.cost_known is False)
            check("单价缺失时成本记 0", no_price_result.cost_cny == 0.0, f"实际 {no_price_result.cost_cny}")

            # ---------------------------------------------------- JSON 模式开关
            print("\n[② 模型层 · 厂商不支持 JSON 模式时]")
            plain_client, _ = build_client(server, json_mode_supported=False, ledger=ledger, budget=budget)
            plain_client.chat([{"role": "user", "content": "不支持就别说支持"}], json_mode=True)
            sent = server.requests[-1]
            check(
                "厂商不支持时不发 response_format",
                "response_format" not in sent["body"],
                f"实际 {sent['body'].get('response_format')}",
            )

            # ---------------------------------------------------- 返回形状
            print("\n[② 模型层 · 畸形返回的处理]")
            obj, err = (None, None)
            fenced = client.chat([{"role": "user", "content": "TRIGGER_FENCED_JSON"}])
            check("围栏 JSON 能拿到原文", "```json" in fenced.text)

            import src.task as task_module

            parsed, parse_error = task_module.parse_json(fenced.text)
            check("被围栏包住的 JSON 仍然能解析出来", parsed is not None and parse_error is None, f"err={parse_error}")
            check("解析结果里有分类字段", isinstance(parsed, dict) and parsed.get("category") == "模型与技术")

            listed = client.chat([{"role": "user", "content": "TRIGGER_CONTENT_LIST"}])
            check("内容分块返回时能拼起来", listed.text == "第一块第二块", f"实际 {listed.text!r}")

            no_usage = client.chat([{"role": "user", "content": "TRIGGER_NO_USAGE"}])
            check("没有 usage 字段时不崩，token 记 0", no_usage.total_tokens == 0, f"实际 {no_usage.total_tokens}")

            expect_raises(
                "没有 choices 时抛致命错误而不是静默返回空",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_NO_CHOICES"}]),
                contains="choices",
            )
            expect_raises(
                "返回不是 JSON 时抛致命错误",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_BAD_JSON"}]),
            )

            # ---------------------------------------------------- 重试
            print("\n[② 模型层 · 失败分类与重试]")

            shim.slept.clear()
            retried = client.chat([{"role": "user", "content": "TRIGGER_RETRY_500_ONCE"}])
            check("500 之后会重试并成功", "模拟摘要内容" in retried.text)
            check("重试后的 attempts 是 2", retried.attempts == 2, f"实际 {retried.attempts}")
            check("退避真的发生了", len(shim.slept) >= 1, f"实际 slept={shim.slept}")
            check("退避时长在合理区间", all(0 <= s <= 25 for s in shim.slept), f"实际 {shim.slept}")

            shim.slept.clear()
            retried_429 = client.chat([{"role": "user", "content": "TRIGGER_RETRY_429"}])
            check("429 之后会重试并成功", "模拟摘要内容" in retried_429.text)
            check("遵守 Retry-After 头", len(shim.slept) >= 1)

            expect_raises(
                "持续 500 会耗尽重试并抛出可重试错误",
                RetryableLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_RETRY_500_ALWAYS"}], retries=2),
                contains="重试",
            )

            # 4xx 不该重试 —— 这是最容易写反的一条
            before = len(server.requests)
            expect_raises(
                "401 是致命错误，不重试",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_FATAL_401"}]),
                contains="密钥",
            )
            check("401 只发了一次请求（没有重试）", len(server.requests) - before == 1,
                  f"实际发了 {len(server.requests) - before} 次")

            before = len(server.requests)
            expect_raises(
                "404 给出可操作的指引",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_FATAL_404"}]),
                contains="probe.py",
            )
            check("404 只发了一次请求（没有重试）", len(server.requests) - before == 1,
                  f"实际发了 {len(server.requests) - before} 次")

            # ---------------------------------------------------- 密钥泄漏
            print("\n[⑤ 权限层 · 密钥不外泄]")
            error = expect_raises(
                "401 的错误信息里带密钥指纹",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_FATAL_401"}]),
            )
            text = str(error or "")
            check("错误信息里没有密钥明文", MOCK_KEY not in text, f"泄漏了：{text[:120]}")
            check("错误信息里保留了指纹（末 4 位）", "9876" in text, f"实际 {text[:160]}")

            # ---------------------------------------------------- 预算
            print("\n[⑤ 权限层 · 预算闸门]")
            tight_budget = Budget(
                {"budget": {"max_calls_per_run": 2, "max_cost_per_run_cny": 100, "max_cost_per_day_cny": 100}}
            )
            tight_client, _ = build_client(server, ledger=ledger, budget=tight_budget)
            tight_client.chat([{"role": "user", "content": "第一次"}])
            tight_client.chat([{"role": "user", "content": "第二次"}])
            expect_raises(
                "超出调用上限后拦住下一次",
                BudgetExceeded,
                lambda: tight_client.chat([{"role": "user", "content": "第三次"}]),
                contains="调用次数",
            )

            # ---------------------------------------------------- 模型发现
            print("\n[② 模型层 · 列模型]")
            models = client.list_models()
            check("能列出模型", "mock-model-1" in models, f"实际 {models}")
            check("列出的模型是排序去重的", models == sorted(set(models)), f"实际 {models}")

            # ---------------------------------------------------- 真实任务链路
            print("\n[ⓞ→② 端到端 · 任务定义 → 消息 → 模型 → 校验]")
            task_cfg = json.loads((PROJECT_ROOT / "config" / "task.json").read_text(encoding="utf-8"))
            messages = task_module.build_messages(
                task_cfg,
                {"title": "模拟标题", "source": "假服务端", "published": "2025-01-01", "body": "模拟正文"},
            )
            outcome = client.chat(
                messages,
                temperature=0.2,
                max_tokens=1200,
                json_mode=True,
                purpose="mocktest:e2e",
            )
            parsed, parse_error = task_module.parse_json(outcome.text)
            problems, warnings = task_module.validate(task_cfg, parsed) if parsed else (["解析失败"], [])
            check("端到端能产出合法结构", parsed is not None and not problems, f"problems={problems}")
            check("端到端没有提醒级问题", warnings == [], f"warnings={warnings}")

            # ---------------------------------------------------- 整条评估链路
            print("\n[⓪ 评估层 · 整条链路（指标计算 + 审核表）]")

            from src.evaluate import _render_review, run_evaluation, summarize

            eval_rows = [
                {"id": "m1", "title": "分类应该对的一条", "body": "正文甲", "gold": {"category": "模型与技术", "importance": 4}},
                {"id": "m2", "title": "分类应该错的一条", "body": "正文乙", "gold": {"category": "融资并购", "importance": 3}},
                {"id": "m3", "title": "没有真值的一条", "body": "正文丙", "gold": {}},
            ]
            eval_records, eval_truncated = run_evaluation(
                client, task_cfg, eval_rows, params=task_cfg["prompt"]["params"], echo=False
            )
            eval_metrics = summarize(
                eval_records,
                client=client,
                task_cfg=task_cfg,
                set_path=PROJECT_ROOT / "eval" / "eval_set.jsonl",
                has_placeholder=False,
                truncated=eval_truncated,
            )

            check("评估跑完所有条数", len(eval_records) == 3, f"实际 {len(eval_records)}")
            check("跑完时没有被标记为中断", eval_truncated is False)
            check("格式全部合规", eval_metrics["format_ok"] == 3, f"实际 {eval_metrics['format_ok']}")
            check("只对有真值的条目算准确率", eval_metrics["category_judged"] == 2,
                  f"实际 {eval_metrics['category_judged']}")
            check("准确率算对了（1 对 1 错 = 50%）", eval_metrics["category_accuracy"] == 0.5,
                  f"实际 {eval_metrics['category_accuracy']}")
            check("重要性平均误差算对了（|4-4| 与 |4-3| 的均值 = 0.5）",
                  eval_metrics["importance_mae"] == 0.5, f"实际 {eval_metrics['importance_mae']}")
            check("无真值的条目不计入准确率分母",
                  eval_records[2].get("category_correct") is None)
            check("成本合计是三条之和",
                  abs(eval_metrics["cost_total_cny"] - 0.0006) < 1e-9,
                  f"实际 {eval_metrics['cost_total_cny']}")

            review = _render_review(eval_records, eval_metrics, "test")
            check("审核表列出了每一条", all(f"`{r['id']}`" in review for r in eval_records))
            check("审核表包含三个必答问题", review.count("原文没说过的话吗") == 3,
                  f"实际出现 {review.count('原文没说过的话吗')} 次")
            check("完整跑完时审核表不写中断警告", "被预算中断" not in review)

            # ---------------------------------------------------- 预算中断
            print("\n[⑤ 权限层 · 预算中断时不丢数据]")

            tiny_budget = Budget(
                {"budget": {"max_calls_per_run": 1, "max_cost_per_run_cny": 100, "max_cost_per_day_cny": 100}}
            )
            tiny_client, _ = build_client(server, ledger=ledger, budget=tiny_budget)

            cut_records, cut_truncated = run_evaluation(
                tiny_client, task_cfg, eval_rows, params=task_cfg["prompt"]["params"], echo=False
            )
            check("预算耗尽时中止并标记为中断", cut_truncated is True)
            check("中断前已完成的结果被保留下来", len(cut_records) == 1, f"实际 {len(cut_records)} 条")
            check("保留下来的那条数据是完整的",
                  cut_records and cut_records[0].get("status") == "ok" and cut_records[0].get("output"),
                  "中断让已完成的数据丢了")

            cut_metrics = summarize(
                cut_records,
                client=tiny_client,
                task_cfg=task_cfg,
                set_path=PROJECT_ROOT / "eval" / "eval_set.jsonl",
                has_placeholder=False,
                truncated=cut_truncated,
            )
            check("中断状态被写进指标（否则两周后会被当成完整基线）",
                  cut_metrics["truncated"] is True)
            cut_review = _render_review(cut_records, cut_metrics, "test")
            check("中断时审核表顶部有警告", "被预算中断" in cut_review)

            # 这一条最重要：预算耗尽不能把异常穿透到调用方
            expect_raises(
                "预算耗尽的真实异常类型是 BudgetExceeded（不是 LLMError）",
                BudgetExceeded,
                lambda: tiny_client.chat([{"role": "user", "content": "再来一次"}]),
            )

    finally:
        llm_module.time = real_time  # type: ignore[assignment]

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
