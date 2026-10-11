"""第 ② 层 · 模型层

这一层只干四件事，多一件都不要：

    1. 把「消息」发给服务端，拿回「文本」      —— 通话
    2. 失败时按规则重试                       —— 稳定性
    3. 记下 token / 成本 / 延迟               —— 计量
    4. 把这些写进账本                         —— 留痕

它【不】知道什么是"资讯摘要"，也不该知道。
任务的知识属于 ⓞ 任务层和 ③ 工作流层。一旦渗进这一层，换任务时你就得重写模型层 ——
而模型层本该是六层里最不用改的一层。

【为什么要做 Provider 抽象】

国内几家（智谱、通义、Kimi、豆包）现在都提供 OpenAI 兼容接口。
这意味着换厂商 = 改 base_url + model，而不是重写代码。
模型一年换三代；你的代码不该跟着换。

【为什么错误要分类】

    4xx（除了 408/429）  —— 重试一万次也没用，是请求本身错了。立刻失败，并说清哪里错。
    408 / 409 / 425 / 429 / 5xx —— 是对方或网络临时的问题。退避后重试。
    超时 / 连接错误      —— 同上。

把这两类混在一起，你就得到两种坏结果：要么该失败的不失败（浪费时间），
要么该重试的不重试（丢数据）。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import requests

from .config import load_policy, load_providers
from .ledger import Budget, Ledger
from .secrets import MissingKey, fingerprint, get_key, mask

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(RuntimeError):
    """模型层错误的基类。"""


class FatalLLMError(LLMError):
    """重试没有意义的错误。"""


class RetryableLLMError(LLMError):
    """临时的错误，值得退避重试。"""


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    cost_cny: float = 0.0
    cost_known: bool = False
    attempts: int = 1
    request_id: str = ""
    json_mode_dropped: bool = False

    def as_ledger_fields(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "cost_cny": self.cost_cny,
            "cost_known": self.cost_known,
            "attempts": self.attempts,
            "request_id": self.request_id,
            # 记下"这家不接受 response_format，被自动降级了"。
            # 结果是好的，但这件事值得留下痕迹：它意味着这家厂商
            # 并不是它声称的那个"完全兼容"，你以后换模型时要心里有数。
            "json_mode_dropped": self.json_mode_dropped,
        }


def autodetect_provider(providers: dict) -> str:
    """在你已有的密钥里挑一个能用的厂商。

    存在的原因很实际：你不一定记得自己申请过哪家的 key。
    与其让你去翻控制台，不如让程序看一眼 .env 然后告诉你。
    """
    available = [
        name
        for name, cfg in providers.items()
        if get_key(cfg.get("api_key_env", ""), required=False)
    ]
    if not available:
        wanted = "\n".join(
            f"    {cfg.get('api_key_env'):<20} {cfg.get('label')}" for cfg in providers.values()
        )
        raise MissingKey(
            "没有找到任何可用的模型密钥。\n"
            f"请在 .env 里填上你实际拥有的一家的 key：\n{wanted}\n"
            "（只有一家也行，其余留空。）"
        )
    return available[0]


class LLMClient:
    """OpenAI 兼容协议的模型客户端。一份代码，多厂商。"""

    def __init__(
        self,
        provider: str | None = None,
        *,
        model: str | None = None,
        providers_cfg: dict | None = None,
        policy_cfg: dict | None = None,
        ledger: Ledger | None = None,
        budget: Budget | None = None,
        require_model: bool = True,
    ) -> None:
        self.providers_cfg = providers_cfg or load_providers()
        self.policy = policy_cfg or load_policy()

        providers = self.providers_cfg.get("providers", {})
        self.provider_name = provider or autodetect_provider(providers)
        if self.provider_name not in providers:
            raise FatalLLMError(
                f"providers.json 里没有名为 {self.provider_name!r} 的厂商。"
                f"可选：{', '.join(providers)}"
            )

        self.provider = providers[self.provider_name]
        self.model = model or self.provider.get("default_model")
        if not self.model and require_model:
            # 注意这个错误的形状：它不只说"失败了"，它告诉你去跑哪条命令。
            # 你在 S5 会再次遇到这个原则 —— 报错的价值在于缩短人到解决之间的距离。
            raise FatalLLMError(
                f"厂商 {self.provider_name} 没有配置默认模型。\n"
                f"  跑 `python tools/probe.py --provider {self.provider_name}` 看它到底有哪些模型，"
                f"然后填进 config/providers.json 的 default_model。"
            )

        # 顺序是有意的：先查权限边界，再索取密钥。
        # 一个越界的配置应该在要密码之前就被拦住 ——
        # 这样"我该不该连这里"和"我的密码对不对"是两个独立的问题，不会互相掩盖。
        self.base_url = str(self.provider.get("base_url", "")).rstrip("/")
        net = self.policy.get("network", {})
        self.timeout = float(net.get("timeout_seconds", 60))
        self.max_retries = int(net.get("max_retries", 3))
        self._enforce_host_allowlist()

        self.api_key = get_key(self.provider.get("api_key_env", ""))
        self.ledger = ledger if ledger is not None else Ledger(self.policy)
        self.budget = budget if budget is not None else Budget(self.policy, ledger=self.ledger)
        self.session = requests.Session()

    # ---------------------------------------------------------------- 权限

    def _enforce_host_allowlist(self) -> None:
        """默认拒绝，而不是默认允许。

        白名单的价值不在于防住攻击者（个人脚本没什么好防的），
        而在于：当某个库偷偷把你的数据发到一个你没意识到的域名时，你会在第一次运行就看到。
        """
        from urllib.parse import urlparse

        host = urlparse(self.base_url).hostname or ""
        allowed = self.policy.get("network", {}).get("allowed_hosts", [])
        if allowed and host not in allowed:
            raise FatalLLMError(
                f"目标域名不在白名单里：{host}\n"
                f"  当前白名单：{', '.join(allowed)}\n"
                f"  确实要访问的话，把它加进 config/policy.json 的 network.allowed_hosts。\n"
                f"  这一步故意设计成手动的 —— 出网这件事应该是有意识的决定。"
            )

    # ---------------------------------------------------------------- 计量

    def _price(self) -> tuple[float, float]:
        price = self.provider.get("price_cny_per_1m") or {}
        inp, out = price.get("input"), price.get("output")
        if inp is None or out is None:
            return 0.0, 0.0
        return float(inp), float(out)

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> tuple[float, bool]:
        inp, out = self._price()
        if inp == 0.0 and out == 0.0:
            # 单价没配就诚实地记 0，并标记 cost_known=False。
            # 假装知道成本，比不知道成本更糟。
            return 0.0, False
        cost = prompt_tokens * inp / 1_000_000 + completion_tokens * out / 1_000_000
        return round(cost, 8), True

    # ---------------------------------------------------------------- 通话

    def _endpoint(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: Sequence[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        retries: int | None = None,
        purpose: str = "",
    ) -> LLMResult:
        """发一次对话请求，失败按规则重试。"""
        use_model = model or self.model
        attempts_allowed = max(1, (retries if retries is not None else self.max_retries))

        payload: dict[str, Any] = {
            "model": use_model,
            "messages": list(messages),
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if json_mode and self.provider.get("supports_json_mode"):
            payload["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        json_mode_dropped = False

        for attempt in range(1, attempts_allowed + 1):
            self.budget.check()
            started = time.perf_counter()
            try:
                response = self.session.post(
                    self._endpoint("chat/completions"),
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = RetryableLLMError(f"网络层失败：{type(exc).__name__}: {exc}")
                self._sleep_backoff(attempt, attempts_allowed)
                continue
            except requests.RequestException as exc:
                raise LLMError(f"请求发送失败：{type(exc).__name__}: {exc}") from exc

            # ---- 厂商不接受 response_format 时，去掉它再试一次
            #
            # "OpenAI 兼容"是各家自己声称的，实际支持的参数并不一致。
            # 有的厂商（或它下面的某些模型）看到 response_format 直接回 400。
            #
            # 而这件事用户完全无从判断：他照着提示做对了每一步，
            # 得到的却是一句"HTTP 400"，而真正的解法是"去掉一个参数"。
            #
            # 好消息是去掉它是安全的：任务的系统提示词里本来就写着
            # "只输出一个 JSON 对象"，所以即使没有 response_format，
            # 模型通常照样会输出 JSON —— 真输出不了，也会被解析器和校验拦下来，
            # 而不是悄悄通过。
            if response.status_code == 400 and "response_format" in payload:
                payload.pop("response_format", None)
                json_mode_dropped = True
                try:
                    response = self.session.post(
                        self._endpoint("chat/completions"),
                        headers=self._headers(),
                        json=payload,
                        timeout=self.timeout,
                    )
                except (requests.Timeout, requests.ConnectionError) as exc:
                    last_error = RetryableLLMError(f"网络层失败：{type(exc).__name__}: {exc}")
                    self._sleep_backoff(attempt, attempts_allowed)
                    continue
                except requests.RequestException as exc:
                    raise LLMError(f"请求发送失败：{type(exc).__name__}: {exc}") from exc

            latency_ms = int((time.perf_counter() - started) * 1000)

            if response.status_code != 200:
                error = self._build_http_error(response)
                if isinstance(error, RetryableLLMError):
                    last_error = error
                    self._sleep_backoff(attempt, attempts_allowed, response=response)
                    continue
                raise error

            try:
                data = response.json()
            except ValueError as exc:
                raise FatalLLMError(
                    f"服务端返回的不是 JSON（HTTP 200）：{mask(response.text[:300])}"
                ) from exc

            return self._finalize(data, use_model, latency_ms, attempt, purpose, json_mode_dropped)

        raise RetryableLLMError(
            f"重试 {attempts_allowed} 次后仍然失败：{last_error}\n"
            f"  可能是网络不稳定，或这家服务当前不可用。"
        )

    def _finalize(
        self,
        data: dict,
        model: str,
        latency_ms: int,
        attempt: int,
        purpose: str,
        json_mode_dropped: bool = False,
    ) -> LLMResult:
        text = self._extract_text(data)
        usage = data.get("usage") or {}
        p_tok = int(usage.get("prompt_tokens") or 0)
        c_tok = int(usage.get("completion_tokens") or 0)
        t_tok = int(usage.get("total_tokens") or (p_tok + c_tok))
        cost, known = self._cost(p_tok, c_tok)

        result = LLMResult(
            text=text,
            provider=self.provider_name,
            model=model,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            total_tokens=t_tok,
            latency_ms=latency_ms,
            cost_cny=cost,
            cost_known=known,
            attempts=attempt,
            request_id=str(data.get("id") or ""),
            json_mode_dropped=json_mode_dropped,
        )

        self.budget.commit(cost=cost, tokens=t_tok)
        self.ledger.record("llm_call", purpose=purpose, **result.as_ledger_fields())
        return result

    @staticmethod
    def _extract_text(data: dict) -> str:
        """兼容几种返回形状。别信"标准协议"，要信真实返回。"""
        choices = data.get("choices") or []
        if not choices:
            raise FatalLLMError(f"返回里没有 choices：{mask(json.dumps(data, ensure_ascii=False)[:300])}")
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        finish_reason = choice.get("finish_reason")

        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            # 有的厂商把内容拆成块
            parts = []
            for block in content:
                if isinstance(block, dict) and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            if "".join(parts).strip():
                return "".join(parts)
        if isinstance(choice.get("text"), str) and choice["text"].strip():
            return choice["text"]

        # ---- 空内容：必须分清是哪一种空，因为解法完全不同
        #
        # 最常见、也最容易被误判的一种：推理模型把 max_tokens 全花在"思考"上，
        # 还没来得及输出答案就被截断了。这时 finish_reason 是 "length"。
        #
        # 如果不区分，用户看到的只会是"输出为空"或者干脆是一堆
        # "格式不合规"，然后他会去怀疑提示词、怀疑模型、怀疑自己 ——
        # 而真正的解法是"把 max_tokens 调大"，报错里一个字都没提。
        if finish_reason == "length":
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            hint = (
                "  模型没说完就停了：输出版额被 max_tokens 用光了。\n"
                "  如果这是个带「思考」的模型，它多半把额度全花在思考上，"
                "还没轮到最后那个 JSON 就被截断了。\n"
            )
            if reasoning:
                hint += f"  （它确实产出了 {len(str(reasoning))} 字的思考内容，但那是给你看的，不是答案。）\n"
            hint += (
                "  改 config/task.json 里的 prompt.params.max_tokens，调大一点（比如 3000）；\n"
                "  或者换一个不带思考的模型。"
            )
            raise FatalLLMError(hint)

        raise FatalLLMError(
            "模型返回了空内容（HTTP 200，但 choices 里没有任何文字）。\n"
            f"  finish_reason={finish_reason!r}\n"
            "  可能是这家服务的临时问题，重跑一次通常就好；"
            "如果每次都这样，换一个模型试试。"
        )

    def _build_http_error(self, response: requests.Response) -> LLMError:
        body = mask(response.text[:500])
        detail = ""
        try:
            parsed = response.json()
            detail = str(
                (parsed.get("error") or {}).get("message")
                or parsed.get("message")
                or parsed.get("msg")
                or ""
            )
        except ValueError:
            pass
        detail = mask(detail)

        hint = ""
        if response.status_code == 401:
            hint = (
                f"\n  密钥被拒绝。常见原因：key 填错、key 已过期、或 key 不属于这个厂商。\n"
                f"  当前使用的密钥指纹：{fingerprint(self.api_key)}"
            )
        elif response.status_code == 403:
            hint = "\n  权限不足：这个 key 可能没开通该模型，或者账号欠费。"
        elif response.status_code == 404:
            hint = (
                f"\n  模型名可能不存在：{self.model}\n"
                f"  先跑 `run.cmd probe --provider {self.provider_name}`，让程序去问服务端有哪些模型。\n"
                f"  如果那家不提供模型列表，就去它的控制台看模型名，"
                f"填进 config/providers.json 的 default_model。"
            )
        elif response.status_code == 429:
            hint = "\n  触发限流或余额不足。"

        message = f"HTTP {response.status_code} 来自 {self.provider_name}：{detail or body}{hint}"

        if response.status_code in RETRYABLE_STATUS:
            return RetryableLLMError(message)
        return FatalLLMError(message)

    @staticmethod
    def _sleep_backoff(attempt: int, total: int, response: requests.Response | None = None) -> None:
        """指数退避 + 抖动。

        抖动（jitter）不是装饰：没有它，多个并发任务会在同一时刻一起重试，
        把对方刚缓过来的服务再打一次。个人项目里养成这个习惯，成本为零。
        """
        if attempt >= total:
            return
        delay = min(2 ** (attempt - 1) * 1.5, 20.0)
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = max(delay, float(retry_after))
                except ValueError:
                    pass
        time.sleep(delay + random.uniform(0, 0.5))

    # ---------------------------------------------------------------- 发现

    def list_models(self) -> list[str]:
        """问服务端它到底有哪些模型。

        这是本项目里"验证优先于记忆"最具体的一处：
        配置文件里的模型名是猜的，这个方法的返回值是真的。
        """
        response = self.session.get(
            self._endpoint("models"), headers=self._headers(), timeout=self.timeout
        )
        if response.status_code != 200:
            raise self._build_http_error(response)
        data = response.json()
        items = data.get("data") or data.get("models") or []
        models: list[str] = []
        for item in items:
            if isinstance(item, str):
                models.append(item)
            elif isinstance(item, dict):
                name = item.get("id") or item.get("name") or item.get("model")
                if name:
                    models.append(str(name))
        return sorted(set(models))

    def ping(self, *, model: str | None = None) -> LLMResult:
        """最小代价的连通性测试：问它一个字。"""
        return self.chat(
            [{"role": "user", "content": "只回复两个字：收到"}],
            model=model,
            temperature=0.0,
            max_tokens=16,
            json_mode=False,
            retries=1,
            purpose="ping",
        )
