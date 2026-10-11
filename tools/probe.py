"""第 ② / ⑤ 层 · 探测：把「我以为」换成「服务端说」

这是本项目里最小的一个工具，却承担着一条最重要的纪律：

    配置文件里的模型名、接口地址、可用性，全都是【猜测】。
    猜测必须被验证。否则你会花一小时去 debug 一个根本不存在的问题。

用法：

    python tools/probe.py                  # 探测所有已配置密钥的厂商
    python tools/probe.py --provider zhipu # 只看一家
    python tools/probe.py --doctor         # 环境自检（不联网）
    python tools/probe.py --check-sources  # 真实抓一次信息源，看哪个是死的
    python tools/probe.py --write-discovered   # 把发现的模型名存档，不自动改配置
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_policy, load_providers, save_json  # noqa: E402
from src.llm import LLMClient  # noqa: E402
from src.secrets import ENV_PATH, PROJECT_ROOT as ROOT, fingerprint, get_key, loaded_secret_count  # noqa: E402

# 明显不是聊天模型的，探测时排除。省得你在一堆 embedding 里找 chat 模型。
NON_CHAT = re.compile(
    r"embed|rerank|image|vision-only|tts|asr|whisper|speech|ocr|video|moderation|guard|safety",
    re.IGNORECASE,
)
# 便宜快速的优先。基线和迭代阶段不需要最贵的模型。
CHEAP_HINT = re.compile(r"flash|mini|turbo|lite|small|air|8k|speed", re.IGNORECASE)


def hr(char: str = "-", width: int = 68) -> str:
    return char * width


# ------------------------------------------------------------------ 探测


def suggest_model(models: list[str], configured: str | None) -> tuple[str | None, bool]:
    """挑一个可用的模型。返回 (模型名, 是否与配置里写的一致)。"""
    if not models:
        return None, False
    if configured and configured in models:
        return configured, True

    chat_models = [m for m in models if not NON_CHAT.search(m)]
    pool = chat_models or models
    cheap = [m for m in pool if CHEAP_HINT.search(m)]
    choice = sorted(cheap)[0] if cheap else sorted(pool)[0]
    return choice, False


def probe_provider(name: str, cfg: dict, policy: dict) -> dict:
    env_name = cfg.get("api_key_env", "")
    key = get_key(env_name, required=False)

    result: dict = {
        "provider": name,
        "label": cfg.get("label", name),
        "env": env_name,
        "has_key": bool(key),
    }
    if not key:
        result["status"] = "no_key"
        return result

    result["key_fingerprint"] = fingerprint(key)

    try:
        client = LLMClient(name, policy_cfg=policy, require_model=False)
    except Exception as exc:  # noqa: BLE001 - 探测就是要吞掉各种异常并报告
        result["status"] = "client_error"
        result["error"] = str(exc)[:300]
        return result

    try:
        models = client.list_models()
        result["models"] = models
    except Exception as exc:  # noqa: BLE001
        result["models"] = []
        result["models_error"] = str(exc)[:200]

    candidate, matched = suggest_model(result.get("models", []), cfg.get("default_model"))
    result["suggested_model"] = candidate
    result["config_matches"] = matched

    if not candidate:
        result["status"] = "no_model_found"
        return result

    try:
        ping = client.ping(model=candidate)
        result["status"] = "ok"
        result["ping_ms"] = ping.latency_ms
        result["ping_text"] = ping.text.strip().replace("\n", " ")[:40]
        result["ping_cost_known"] = ping.cost_known
    except Exception as exc:  # noqa: BLE001
        result["status"] = "ping_failed"
        result["ping_error"] = str(exc)[:400]

    return result


def print_probe_report(results: list[dict]) -> None:
    print()
    print(hr("="))
    print("  探测结果")
    print(hr("="))

    for item in results:
        label = f"{item['label']}  ({item['provider']})"
        print(f"\n[{label}]")

        if item["status"] == "no_key":
            print(f"  跳过：.env 里没有 {item['env']}")
            continue

        print(f"  密钥       {item.get('key_fingerprint', '(未知)')}  来自 {item['env']}")

        if item["status"] == "client_error":
            print(f"  构造客户端失败：{item['error']}")
            continue

        models = item.get("models", [])
        if models:
            print(f"  可用模型   {len(models)} 个")
            for model in models[:14]:
                mark = "  <- 配置里写的" if model == item.get("suggested_model") and item.get("config_matches") else ""
                print(f"               {model}{mark}")
            if len(models) > 14:
                print(f"               ...（还有 {len(models) - 14} 个）")
        else:
            print(f"  可用模型   查不到（{item.get('models_error', '该厂商可能不提供 /models 接口')[:110]}）")
            print("            这不一定是问题：有的厂商只支持直接调用。")

        if item["status"] == "ok":
            print(f"  实际调用   成功  {item['ping_ms']}ms  返回：{item['ping_text']!r}")
            if item.get("suggested_model"):
                note = "（与配置一致）" if item.get("config_matches") else "（与配置不同！请更新）"
                print(f"  建议模型   {item['suggested_model']} {note}")
        elif item["status"] == "ping_failed":
            print(f"  实际调用   失败：{item.get('ping_error')}")
        elif item["status"] == "no_model_found":
            print("  实际调用   跳过：拿不到任何模型名，无法测试。请手动填 config/providers.json 的 default_model。")

    print()
    print(hr("="))
    ready = [r for r in results if r["status"] == "ok"]
    if ready:
        best = ready[0]
        print("  可用的厂商：")
        for item in ready:
            print(f"    {item['provider']:<12} {item.get('suggested_model')}")
        print()
        print("  下一步：")
        print(f"    1) 打开 config/providers.json，把 {best['provider']} 的 default_model 改成")
        print(f"       {best.get('suggested_model')}")
        print("       （顺便把 price_cny_per_1m 填上，成本才有可能被算出来）")
        print("    2) 跑 python src/evaluate.py --limit 3")
    else:
        print("  没有可用的厂商。")
        print("  请检查：")
        print("    a) .env 里是否填了 key（文件名就叫 .env，不是 .env.txt）")
        print("    b) key 是否抄全、有没有多余空格")
        print("    c) 账号是否已实名/欠费")
        print(f"  当前 .env 路径：{ENV_PATH}")
    print(hr("="))
    print()


# ------------------------------------------------------------------ 环境自检


def run_doctor() -> int:
    print()
    print(hr("="))
    print("  环境自检")
    print(hr("="))
    ok = True

    print(f"\n[Python]")
    print(f"  版本       {sys.version.split()[0]}")
    print(f"  解释器     {sys.executable}")
    if sys.version_info < (3, 10):
        print("  ! 建议 3.10 以上")
        ok = False

    print(f"\n[依赖]")
    try:
        import requests

        print(f"  requests   {requests.__version__}")
    except ImportError:
        print("  ! 缺少 requests：python -m pip install requests")
        ok = False

    print(f"\n[目录]")
    for relative in ("config", "src", "tools", "eval", "data", "out/logs"):
        path = ROOT / relative
        mark = "存在" if path.is_dir() else "缺失"
        print(f"  {relative:<12} {mark}")
        if not path.is_dir():
            ok = False

    print(f"\n[密钥文件]")
    if ENV_PATH.exists():
        print(f"  .env       存在   {ENV_PATH}")
        print(f"  已登记密钥 {loaded_secret_count()} 个（只统计看起来像密钥的变量）")
    else:
        print(f"  .env       不存在")
        print(f"  请把 .env.example 复制一份改名为 .env，然后填 key。")
        ok = False

    print(f"\n[配置]")
    try:
        providers = load_providers().get("providers", {})
        policy = load_policy()
        print(f"  providers  {len(providers)} 家：{', '.join(providers)}")
        print(f"  policy     单次限额 {policy.get('budget', {}).get('max_cost_per_run_cny')} 元 / "
              f"{policy.get('budget', {}).get('max_calls_per_run')} 次")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! 配置读取失败：{exc}")
        return 1

    # 白名单一致性：这是最容易被忽略、又最容易卡住的一次性检查
    print(f"\n[白名单一致性]")
    from urllib.parse import urlparse

    allowed = set(policy.get("network", {}).get("allowed_hosts", []))
    for name, cfg in providers.items():
        host = urlparse(str(cfg.get("base_url", ""))).hostname or ""
        if host in allowed:
            print(f"  {name:<12} {host}  已允许")
        else:
            print(f"  {name:<12} {host}  ! 不在白名单，调用会被拒绝")
            ok = False

    print()
    print(hr("="))
    print("  自检通过。" if ok else "  自检发现问题，见上面标 ! 的行。")
    print(hr("="))
    print()
    return 0 if ok else 1


# ------------------------------------------------------------------ 信息源检查


def check_sources() -> int:
    from src.config import load_sources

    config = load_sources()
    sources = [s for s in config.get("sources", []) if s.get("enabled") and s.get("url")]

    print()
    print(hr("="))
    print("  信息源检查（真实抓取）")
    print(hr("="))

    if not sources:
        print("\n  config/sources.json 里没有启用任何源。")
        print("  这是 S2 阶段的事。现在只需要知道：加源的时候，")
        print("  一定要先跑这个命令验证它是活的 —— 一个死掉的源会静默产出空结果。\n")
        return 0

    import requests

    timeout = 20
    failures = 0
    for source in sources:
        name = source.get("name", "(无名)")
        url = source["url"]
        try:
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "info-radar/0.1 (+personal research)"},
            )
            body = response.text
            entries = len(re.findall(r"<(?:item|entry)[\s>]", body))
            if response.status_code == 200 and entries:
                print(f"\n  [OK] {name}")
                print(f"       {response.status_code}  {entries} 条  {len(body)} 字节")
            elif response.status_code == 200:
                print(f"\n  [?]  {name}")
                print(f"       200 但解析不出条目。可能不是 RSS/Atom，而是网页。")
                failures += 1
            else:
                print(f"\n  [!!] {name}")
                print(f"       HTTP {response.status_code}")
                failures += 1
        except Exception as exc:  # noqa: BLE001
            print(f"\n  [!!] {name}")
            print(f"       {type(exc).__name__}: {str(exc)[:150]}")
            failures += 1

    print()
    print(hr("="))
    print(f"  成功 {len(sources) - failures} / {len(sources)}")
    print(hr("="))
    print()
    return 0 if failures == 0 else 1


# ------------------------------------------------------------------ 入口


def main() -> int:
    parser = argparse.ArgumentParser(description="探测模型接口真实可用性")
    parser.add_argument("--provider", "-Provider", default=None, help="只探测这一家")
    parser.add_argument("--doctor", action="store_true", help="环境自检，不联网")
    parser.add_argument("--check-sources", action="store_true", help="真实抓取信息源")
    parser.add_argument("--write-discovered", action="store_true", help="把发现的模型名存到 config/providers.discovered.json")
    args = parser.parse_args()

    if args.doctor:
        return run_doctor()
    if args.check_sources:
        return check_sources()

    providers = load_providers().get("providers", {})
    policy = load_policy()

    if args.provider:
        if args.provider not in providers:
            print(f"没有名为 {args.provider!r} 的厂商。可选：{', '.join(providers)}")
            return 2
        targets = {args.provider: providers[args.provider]}
    else:
        targets = providers

    print(f"\n正在探测 {len(targets)} 家厂商……（会真实发请求，耗时几秒到几十秒）")
    results = [probe_provider(name, cfg, policy) for name, cfg in targets.items()]

    print_probe_report(results)

    if args.write_discovered:
        payload = {
            "_note": "由 tools/probe.py 生成。这是服务端的真实回答，不是猜测。",
            "discovered": {
                item["provider"]: {
                    "models": item.get("models", []),
                    "suggested_model": item.get("suggested_model"),
                    "config_matches": item.get("config_matches"),
                    "status": item["status"],
                }
                for item in results
            },
        }
        path = save_json("providers.discovered", payload)
        print(f"已存档：{path}\n")

    return 0 if any(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[中断]\n")
        raise SystemExit(130)
