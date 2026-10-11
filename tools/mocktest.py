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
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

import src.llm as llm_module  # noqa: E402
import src.collect as collect_mod  # noqa: E402
from probe import suggest_model  # noqa: E402
from src.pipeline import StubClient  # noqa: E402
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
            # 有的厂商压根没有这个接口。用 state 开关模拟，
            # 因为"列不出模型但密钥是好的"是一条会让人第一次就卡死的路径。
            if self.server.state.get("models_404"):  # type: ignore[attr-defined]
                self._send(404, {"error": {"message": "no such endpoint"}})
                return
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

        if "TRIGGER_REJECT_JSON_MODE" in text and "response_format" in body:
            # "OpenAI 兼容"是各家自己声称的，实际支持的参数并不一致。
            self._send(
                400,
                {"error": {"message": "invalid parameter: response_format is not supported"}},
            )
            return

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

        if "TRIGGER_EMPTY_LENGTH" in text:
            # 推理模型把 max_tokens 全花在思考上，还没来得及给答案就被截断。
            # 这是最容易被误判成"提示词不好"或"模型不行"的一种失败。
            self._send(
                200,
                {
                    "id": "chatcmpl-empty",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "让我想想……先分析一下这篇文章讲了什么……" * 20,
                            },
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 1200},
                },
            )
            return

        if "TRIGGER_EMPTY_OTHER" in text:
            self._send(
                200,
                {
                    "id": "chatcmpl-empty2",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                    "usage": {},
                },
            )
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

            # 空内容分两种，解法完全不同。混在一起报，用户会去怀疑提示词和模型，
            # 而真正该做的是把 max_tokens 调大。
            print("\n[② 模型层 · 模型返回空内容时]")
            truncated = expect_raises(
                "被 max_tokens 截断（推理模型常见）时，明说该调 max_tokens",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_EMPTY_LENGTH"}]),
                contains="max_tokens",
            )
            check("截断时的提示指出它把额度花在了思考上",
                  "思考" in str(truncated or ""), f"实际 {str(truncated or '')[:120]}")
            check("截断时的提示给出具体的修改位置",
                  "config/task.json" in str(truncated or ""))

            other_empty = expect_raises(
                "别的原因导致空内容时，给的是另一种解释",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_EMPTY_OTHER"}]),
                contains="空内容",
            )
            check("两种空内容的提示确实不同",
                  "max_tokens" not in str(other_empty or ""),
                  "两种情况混成一句话，就等于没诊断")

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
            not_found = expect_raises(
                "404 给出可操作的指引",
                FatalLLMError,
                lambda: client.chat([{"role": "user", "content": "TRIGGER_FATAL_404"}]),
                contains="run.cmd probe",
            )
            check("404 的提示里给了「这家不提供模型列表」时的退路",
                  "控制台" in str(not_found or ""),
                  f"实际 {str(not_found or '')[:140]}")
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

            # 有的厂商不提供 /models 接口。那和"密钥好不好"毫无关系 ——
            # 绝不能因此告诉用户"无法测试"，那会让他以为自己的密钥有问题。
            print("\n[② 模型层 · 厂商不提供模型列表时]")
            server.httpd.state["models_404"] = True  # type: ignore[attr-defined]
            try:
                expect_raises(
                    "厂商没有 /models 接口时，列模型会失败",
                    FatalLLMError,
                    lambda: client.list_models(),
                )
                fallback, matched = suggest_model([], "mock-model-1")
                check("列不出模型时，退回用配置里写的那个",
                      fallback == "mock-model-1" and matched is True, f"实际 {fallback!r}")
                recovered = client.chat(
                    [{"role": "user", "content": "配置里的模型名还能用吗"}], model=fallback
                )
                check("用配置里的模型名照样能正常调用",
                      "模拟摘要内容" in recovered.text,
                      "密钥明明是好的，却因为列不出模型而报错 —— 这正是要避免的死局")
            finally:
                server.httpd.state["models_404"] = False  # type: ignore[attr-defined]

            # 有的厂商（或它下面的某些模型）看到 response_format 直接回 400。
            # 用户完全无从判断：他每一步都做对了，拿到的却是一句 HTTP 400，
            # 而真正的解法是"去掉一个参数"。
            print("\n[② 模型层 · 厂商拒绝 response_format 时]")
            before = len(server.requests)
            recovered = client.chat(
                [{"role": "user", "content": "TRIGGER_REJECT_JSON_MODE 给我结果"}],
                json_mode=True,
                purpose="mocktest:jsonfallback",
            )
            check("厂商拒绝 response_format 时会自动去掉它再试一次",
                  "模拟摘要内容" in recovered.text, f"实际 {recovered.text[:70]!r}")
            check("这件事被记在结果里，不是悄悄吞掉",
                  recovered.json_mode_dropped is True,
                  "自动降级必须留痕 —— 否则你以后换模型时不知道这家并不完全兼容")
            check("确实发了两次请求（一次带参数、一次不带）",
                  len(server.requests) - before == 2,
                  f"实际 {len(server.requests) - before} 次")
            retry_body = server.requests[-1]["body"]
            check("重试那次真的去掉了 response_format",
                  "response_format" not in retry_body, f"实际 {retry_body.get('response_format')}")
            check("重试没有丢掉别的参数",
                  retry_body.get("model") == "mock-model-1" and bool(retry_body.get("messages")),
                  "去掉一个参数不该顺手把请求改坏")
            ledger_rows = ledger.read_today()
            check("账本里记下了这次降级",
                  any(r.get("json_mode_dropped") for r in ledger_rows),
                  "账本是审计用的，降级这种事必须在里面")

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

            # 占位样本不能计入准确率。
            # 它们是合成内容（"甲实验室""乙公司"），拿真实文章去测，
            # 模型永远不可能命中它们 —— 算进分母只会得到一个被凭空拉低的假数字。
            # 而一个假数字比没有数字更坏：你会照着它去优化。
            mixed_records = [
                {"id": "ph1", "placeholder": True, "format_ok": True, "status": "ok",
                 "category_correct": False, "importance_error": 1, "latency_ms": 1,
                 "cost_cny": 0.0, "cost_known": True, "prompt_tokens": 1, "completion_tokens": 1},
                {"id": "r1", "placeholder": False, "format_ok": True, "status": "ok",
                 "category_correct": True, "importance_error": 0, "latency_ms": 1,
                 "cost_cny": 0.0, "cost_known": True, "prompt_tokens": 1, "completion_tokens": 1},
                {"id": "r2", "placeholder": False, "format_ok": True, "status": "ok",
                 "category_correct": False, "importance_error": 2, "latency_ms": 1,
                 "cost_cny": 0.0, "cost_known": True, "prompt_tokens": 1, "completion_tokens": 1},
            ]
            mixed_metrics = summarize(
                mixed_records,
                client=StubClient(),
                task_cfg=task_cfg,
                set_path=PROJECT_ROOT / "eval" / "eval_set.jsonl",
                has_placeholder=True,
            )
            check("占位样本被单独计数", mixed_metrics["placeholder_count"] == 1,
                  f"实际 {mixed_metrics['placeholder_count']}")
            check("真正计分的样本数不含占位", mixed_metrics["scored"] == 2,
                  f"实际 {mixed_metrics['scored']}")
            check("准确率只按非占位样本算（1/2 = 50%，不是 1/3）",
                  mixed_metrics["category_accuracy"] == 0.5,
                  f"实际 {mixed_metrics['category_accuracy']} —— 把合成条目算进分母会凭空拉低它")
            check("格式合规率仍然算全部（格式对错和内容真假无关）",
                  mixed_metrics["format_ok"] == 3, f"实际 {mixed_metrics['format_ok']}")

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

            # ---------------------------------------------------- 工作流层
            print("\n[③ 工作流层 · 编排 / 失败隔离 / 断点续跑]")

            from src.pipeline import Pipeline

            def make_env(where: Path, bodies: dict[str, str]):
                """在临时目录里造一个完整的 pipeline 环境：原文 + 索引。"""
                raw_dir = where / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                index_path = where / "index.json"
                results_dir = where / "results"
                items = {}
                for order, (item_id, body) in enumerate(bodies.items()):
                    record = {
                        "id": item_id,
                        "title": f"条目 {item_id}",
                        "source": "测试源",
                        "body": body,
                        "body_chars": len(body),
                        "fetched_at": f"2025-01-{order + 1:02d}T00:00:00+08:00",
                        "content_hash": item_id,
                    }
                    (raw_dir / f"{item_id}.json").write_text(
                        json.dumps(record, ensure_ascii=False), encoding="utf-8"
                    )
                    items[item_id] = {
                        "title": record["title"],
                        "source": "测试源",
                        "processed": False,
                        "status": None,
                        "attempts": 0,
                    }
                index_path.write_text(
                    json.dumps({"items": items, "content_hashes": {}}, ensure_ascii=False),
                    encoding="utf-8",
                )
                return raw_dir, index_path, results_dir

            def read_index(path: Path) -> dict:
                return json.loads(path.read_text(encoding="utf-8"))["items"]

            # ---- 失败隔离
            env_dir = Path(tmp) / "wf"
            raw_dir, index_path, results_dir = make_env(
                env_dir,
                {
                    "w1": "一条完全正常的正文。",
                    "w2": "TRIGGER_CONTENT_LIST",          # 返回非 JSON -> 格式不合规
                    "w3": "TRIGGER_RETRY_500_ALWAYS",      # 持续 500 -> 调用失败
                    "w4": "另一条完全正常的正文。",
                },
            )
            wf_client, _ = build_client(server, ledger=ledger, budget=budget)
            wf = Pipeline(
                wf_client, task_cfg, raw_dir=raw_dir, index_path=index_path,
                results_dir=results_dir, echo=False,
            )
            items = wf.select()
            check("工作流层能选出待处理条目", len(items) == 4, f"实际 {len(items)}")

            run1, path1 = wf.run(items)
            check("正常条目成功", run1.ok_count == 2, f"实际 {run1.ok_count}")
            check("非 JSON 返回被记为格式不合规", run1.format_error_count == 1,
                  f"实际 {run1.format_error_count}")
            check("持续失败的条目被记为调用失败", run1.call_failed_count == 1,
                  f"实际 {run1.call_failed_count}")
            check("一条失败没有拖垮整批（其余三条都跑完了）", len(run1.outcomes) == 4,
                  f"实际 {len(run1.outcomes)}")
            check("产出文件写出来了", path1 is not None and path1.exists())

            summary = json.loads(
                (results_dir / f"{path1.stem}.summary.json").read_text(encoding="utf-8")
            )
            check("汇总文件记下了失败分类",
                  summary["call_failed"] == 1 and summary["format_error"] == 1,
                  f"实际 {summary}")

            # ---- 断点续跑
            index_now = read_index(index_path)
            check("成功条目被标记为已处理",
                  index_now["w1"]["processed"] is True and index_now["w4"]["processed"] is True)
            check("失败条目也被标记为已处理（否则每次运行都会重试、每次扣钱）",
                  index_now["w3"]["processed"] is True, "失败条目没有被标记，会造成静默烧钱")
            check("失败条目留下了可读的失败原因", bool(index_now["w3"].get("last_error")))
            check("失败条目的状态被记下来（不是靠日志去猜）",
                  index_now["w3"]["status"] == "call_failed", f"实际 {index_now['w3']['status']}")

            wf2 = Pipeline(
                wf_client, task_cfg, raw_dir=raw_dir, index_path=index_path,
                results_dir=results_dir, echo=False,
            )
            check("重跑时不再选出任何条目（这就是断点续跑，也是不重复花钱）",
                  wf2.select() == [], f"实际选出 {len(wf2.select())} 条")

            retry_items = wf2.select(retry_failed=True)
            # 名字要说准：不是「只挑出失败的」，而是
            # 「在已处理的里面，只有失败的会被重新选出」。
            # 本场景里 4 条全都处理过了，所以看起来一样；
            # 真实场景里还有一堆没跑过的，它们照样会在列表里。
            check("--retry-failed 时，已处理的条目里只有失败的会被重新选出",
                  len(retry_items) == 2, f"实际 {len(retry_items)} 条")

            # ---- --retry-failed 到底是什么意思
            #
            # 每个人第一眼都会把它读成「只重试失败的那几条」。
            # 而实际行为是「把失败的也捞回来，和没跑过的混在一起」。
            #
            # 这个误解会花钱：你以为要处理 2 条，实际处理 116 条 —— 差 50 倍。
            # 所以 is_retry 必须能被单独问清楚，而不是只在输出里算个数字。
            check("is_retry 能认出「处理过但失败了」的条目",
                  wf2.is_retry("w3") is True,
                  f"w3 的状态是 {index_now['w3'].get('status')}，应被认成重试")
            check("is_retry 不会把「成功过的」算成重试",
                  wf2.is_retry("w1") is False,
                  "成功的被算成重试，那个数字就虚高了")
            check("is_retry 不会把「从来没跑过的」算成重试",
                  wf2.is_retry("根本没这个 id") is False,
                  "把没跑过的也算进来，用户会以为失败了一大片")

            # ---- 预算中断
            budget_dir = Path(tmp) / "wf_budget"
            raw_dir_b, index_path_b, results_dir_b = make_env(
                budget_dir, {f"b{i}": f"正常正文第 {i} 条。" for i in range(1, 5)}
            )
            cut_budget = Budget(
                {"budget": {"max_calls_per_run": 2, "max_cost_per_run_cny": 100, "max_cost_per_day_cny": 100}}
            )
            cut_client, _ = build_client(server, ledger=ledger, budget=cut_budget)
            wf3 = Pipeline(
                cut_client, task_cfg, raw_dir=raw_dir_b, index_path=index_path_b,
                results_dir=results_dir_b, echo=False,
            )
            run3, _ = wf3.run(wf3.select())
            check("预算中断时标记 truncated", run3.truncated is True)
            check("中断时停在中途（只处理了 2 条）", len(run3.outcomes) == 2,
                  f"实际 {len(run3.outcomes)}")

            index_b = read_index(index_path_b)
            done = [k for k, v in index_b.items() if v.get("processed")]
            todo = [k for k, v in index_b.items() if not v.get("processed")]
            check("已处理的 2 条被标记", len(done) == 2, f"实际 {done}")
            check("没碰过的条目保持未处理状态（下次会接着跑）", len(todo) == 2, f"实际 {todo}")

            wf4 = Pipeline(
                cut_client, task_cfg, raw_dir=raw_dir_b, index_path=index_path_b,
                results_dir=results_dir_b, echo=False,
            )
            check("中断后重跑，只剩下没处理的那 2 条",
                  len(wf4.select()) == 2, f"实际 {len(wf4.select())} 条")

            # ---- 假模型
            print("\n[③ 工作流层 · 没有密钥时的空转模式]")

            stub_dir = Path(tmp) / "wf_stub"
            raw_dir_s, index_path_s, results_dir_s = make_env(stub_dir, {"s1": "一段正文。"})
            stub = StubClient()
            wf5 = Pipeline(
                stub, task_cfg, raw_dir=raw_dir_s, index_path=index_path_s,
                results_dir=results_dir_s, echo=False,
            )
            run5, _ = wf5.run(wf5.select())
            check("假模型下链路照样跑通", run5.ok_count == 1, f"实际 {run5.ok_count}")
            check("假模型的输出带明显标记（假结果看起来像真的最危险）",
                  "[假模型]" in json.dumps(run5.outcomes[0].output, ensure_ascii=False))
            check("假模型不产生成本", run5.total_cost == 0.0, f"实际 {run5.total_cost}")

            # 评估层也要有假模型通道。
            # 之前只有 pipeline 和 daily 有 --stub，eval 没有 —— 结果就是
            # 【你唯一要花钱的那一步，偏偏是唯一不能预览的那一步】。
            # 现在补上了，这里守一条回归：假模型必须能跑完整条评估链路。
            stub_eval_records, stub_eval_truncated = run_evaluation(
                StubClient(), task_cfg, eval_rows, params=task_cfg["prompt"]["params"], echo=False
            )
            check("假模型能跑完整条评估链路", len(stub_eval_records) == 3,
                  f"实际 {len(stub_eval_records)} 条")
            check("假模型的输出全部结构合规",
                  all(r.get("format_ok") for r in stub_eval_records),
                  "假模型产出了不合结构的 JSON —— 那样 --stub 就验证不了链路")
            check("假模型评估不会误报中断", stub_eval_truncated is False)
            stub_eval_metrics = summarize(
                stub_eval_records,
                client=StubClient(),
                task_cfg=task_cfg,
                set_path=PROJECT_ROOT / "eval" / "eval_set.jsonl",
                has_placeholder=True,
                truncated=False,
            )
            check("假模型评估的成本是 0", stub_eval_metrics["cost_total_cny"] == 0.0)
            check("假模型的准确率算出来了（它是 0，因为假模型永远答「其他」）",
                  stub_eval_metrics["category_accuracy"] == 0.0,
                  f"实际 {stub_eval_metrics['category_accuracy']}")

            # ---------------------------------------------------- 大声失败
            print("\n[⑤ 权限层 · 大声失败的三条保证]")
            import src.pipeline as pipeline_mod
            from src import runner as runner_mod
            from src.runner import StepSkipped, _step_digest, _step_pipeline

            # 1) 全部信息源失败 —— 不该被当成"今天没有新内容"
            dead_sources = {
                "sources": [{"name": "死源", "url": "http://127.0.0.1:1/feed", "enabled": True}],
                "fetch": {"timeout_seconds": 2, "delay_between_sources": 0, "min_body_chars": 10},
            }
            real_load_sources = collect_mod.load_sources
            real_load_policy = collect_mod.load_policy
            # 关键：把索引和状态文件的路径也换到临时目录。
            # 第一版漏了这一步，于是测试里那个"死源"被真的写进了
            # data/cache/sources-state.json，之后每次 run.cmd status 都在报一个不存在的源。
            # **测试绝不能写进活状态。** 一个会污染真实数据的测试，
            # 比没有测试更坏 —— 它会让你开始怀疑真实数据。
            real_index_path = collect_mod.INDEX_PATH
            real_state_path = collect_mod.STATE_PATH
            collect_mod.load_sources = lambda: dead_sources
            collect_mod.load_policy = lambda: {"network": {"feed_hosts": ["127.0.0.1"]}}
            collect_mod.INDEX_PATH = Path(tmp) / "dead_index.json"
            collect_mod.STATE_PATH = Path(tmp) / "dead_state.json"
            try:
                error = expect_raises(
                    "全部信息源失败时大声报错", RuntimeError, runner_mod._step_collect
                )
                check("报错信息里指名道姓地列出了失败的源", "死源" in str(error or ""),
                      f"实际 {str(error or '')[:100]}")
                check("失败的源被记进状态文件（供 run.cmd status 显示）",
                      (Path(tmp) / "dead_state.json").exists(), "状态没落盘")
            finally:
                collect_mod.load_sources = real_load_sources
                collect_mod.load_policy = real_load_policy
                collect_mod.INDEX_PATH = real_index_path
                collect_mod.STATE_PATH = real_state_path

            # 2) 所有条目都调用失败 —— 那是密钥/余额/网络的问题，不是内容的问题
            class FakePipeline:
                def __init__(self, *a, **k):
                    pass

                def select(self, **k):
                    return [{"id": "x1", "title": "会失败的条目"}]

                def run(self, items):
                    from src.pipeline import ItemOutcome, RunResult

                    result = RunResult(candidates=1)
                    result.outcomes.append(
                        ItemOutcome(item_id="x1", status="call_failed", error="boom")
                    )
                    return result, None

            real_pipeline_class = pipeline_mod.Pipeline
            pipeline_mod.Pipeline = FakePipeline
            try:
                expect_raises(
                    "所有条目都调用失败时大声报错，并指出不是内容的问题",
                    RuntimeError,
                    lambda: _step_pipeline(stub=True),
                    contains="不是内容有问题",
                )
            finally:
                pipeline_mod.Pipeline = real_pipeline_class

            # 3) 没有新结果时不能拿旧结果去渲染日报 —— 但也【不能】算失败
            expect_raises(
                "没有新结果时拒绝渲染旧日报",
                StepSkipped,
                lambda: _step_digest(None),
                contains="没有可渲染",
            )
            # 「跳过」与「失败」的完整区分（含计数归零、警报撤销）在 selftest 里，
            # 那里不需要 HTTP，跑得更快，也更适合放这种纯状态机断言。

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
