"""第 ① 层 · 数据层：抓取、归一、去重、缓存

【为什么它在 S1 阶段就先做了一小片】

严格按路线，数据层是 S2。但 S1 要的那 20 条真值样本必须来自【真实】文档，
凭空编的样本只能验证链路，不能测出真实效果。所以这里先做最窄的一刀：
**只做"取到干净的原文"这一件事**，不做处理、不做编排。

这也是这套结构的一个真实特性：分层是【责任划分】，不是施工顺序。
你可以先实现某一层的 10%，只要那 10% 的责任边界是清楚的。

【这一层的四条原则】

1. 优先结构化源（RSS / Atom），不解析网页。
   网页会改版，改版就静默失效；RSS 是契约。能拿契约就不要去猜别人的 HTML。
   实测：机器之心和 36氪的 RSS 现在都返回 HTML —— 这就是"源会死"的样子。

2. 两级去重。
   URL 指纹：同一篇不抓两次。
   内容指纹：换了标题、换了链接的同一篇内容不重复处理。
   第二级才是真正省钱的那一级。

3. 缓存即账本。
   抓下来的原文落盘，处理状态记在数据里。重跑不会重复调 API，也就不会重复付费。

4. 单源失败要报告，全部失败要非零退出。
   一个悄悄死掉的源，比一个报错的源危险得多 —— 它会让你以为"今天没什么新闻"。

用法：
    python src/collect.py                 # 抓取所有启用的源
    python src/collect.py --source 量子位  # 只抓一个源
    python src/collect.py --limit 3        # 每个源最多取 3 条
    python src/collect.py --list           # 只看配置，不联网
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_policy, load_sources  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
INDEX_PATH = CACHE_DIR / "index.json"
STATE_PATH = CACHE_DIR / "sources-state.json"

DEFAULT_FETCH = {
    "timeout_seconds": 20,
    "max_items_per_source": 30,
    "max_body_chars": 3000,
    "min_body_chars": 200,
    "delay_between_sources": 1.0,
    "user_agent": "info-radar/0.1 (+personal research; contact: local)",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\u3000]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


# ------------------------------------------------------------------ 文本处理


def strip_html(text: str) -> str:
    """把 HTML 压成纯文本。

    不做完整的 HTML 解析，因为 RSS 里的 description 通常是简单片段。
    如果哪天发现源里出现了复杂嵌套标签，再升级到解析器 ——
    先用手边最简单的办法，等它真的不够用再换。
    """
    if not text:
        return ""
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|li|h[1-6])>", "\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    return _BLANK_LINES_RE.sub("\n\n", text).strip()


def normalize_url(url: str) -> str:
    """去掉追踪参数。同一个页面带不同 utm 参数，在去重眼里是同一条。"""
    if not url:
        return ""
    url = url.strip()
    if "?" in url:
        base, _, query = url.partition("?")
        kept = [
            pair
            for pair in query.split("&")
            if pair and not pair.split("=")[0].lower().startswith(("utm_", "fbclid", "gclid", "spm", "ref"))
        ]
        url = base + ("?" + "&".join(kept) if kept else "")
    return url


def make_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:12]


def content_hash(title: str, body: str) -> str:
    """内容指纹。只保留字母数字和汉字，忽略标点与空白差异。"""
    normalized = re.sub(r"[^\w\u4e00-\u9fff]+", "", f"{title}{body}").lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ------------------------------------------------------------------ 源解析


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if "}" in tag else tag.lower()


def _child_text(node: ET.Element, names: tuple[str, ...]) -> str:
    """按本地名取子节点文本，忽略命名空间。

    忽略命名空间是刻意的：RSS 2.0 用 item/description，Atom 用 entry/summary，
    content:encoded 又自带一个命名空间。逐个硬编码会让解析器极度脆弱，
    而按本地名匹配能同时吃下这几种。
    """
    for child in node:
        if _local_name(child.tag) in names:
            if child.text and child.text.strip():
                return child.text.strip()
    return ""


def _child_link(node: ET.Element) -> str:
    """链接最麻烦：RSS 放在 <link> 文本里，Atom 放在 <link href="..."> 属性里。"""
    for child in node:
        if _local_name(child.tag) != "link":
            continue
        href = child.get("href")
        if href:
            rel = (child.get("rel") or "alternate").lower()
            if rel == "alternate":
                return href.strip()
            continue
        if child.text and child.text.strip():
            return child.text.strip()
    # 退一步：有的源把链接放在 guid / id 里
    for name in ("guid", "id"):
        value = _child_text(node, (name,))
        if value.startswith("http"):
            return value
    return ""


def parse_feed(xml_bytes: bytes, source_name: str) -> list[dict]:
    """解析 RSS 2.0 或 Atom。返回归一化后的条目列表。"""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise ValueError(f"XML 解析失败：{exc}") from exc

    items: list[dict] = []
    for node in root.iter():
        if _local_name(node.tag) not in ("item", "entry"):
            continue

        title = strip_html(_child_text(node, ("title",)))
        url = normalize_url(_child_link(node))
        # content:encoded 本地名是 encoded；优先用它，它通常是全文
        raw_body = (
            _child_text(node, ("encoded",))
            or _child_text(node, ("content",))
            or _child_text(node, ("description", "summary"))
        )
        body = strip_html(raw_body)
        published = _child_text(node, ("pubdate", "published", "updated", "date"))

        if not title and not body:
            continue

        items.append(
            {
                "title": title,
                "url": url,
                "source": source_name,
                "published": published or None,
                "body": body,
            }
        )
    return items


# ------------------------------------------------------------------ 索引


def load_json_file(path: Path, default: dict) -> dict:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def atomic_write_json(path: Path, payload: dict) -> None:
    """先写临时文件再替换。

    直接覆写有一个真实的坏结局：写到一半断电，你既没有新数据也丢了旧数据。
    原子替换让"要么旧的完整、要么新的完整"成为保证。
    这类小习惯的总成本很低，而它保护的正是你最不想重跑的东西。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_index() -> dict:
    index = load_json_file(INDEX_PATH, {"items": {}, "content_hashes": {}})
    index.setdefault("items", {})
    index.setdefault("content_hashes", {})
    return index


# ------------------------------------------------------------------ 抓取


def fetch_source(source: dict, fetch_cfg: dict, limit: int, session) -> dict:
    name = source.get("name") or source.get("url", "(无名)")
    url = source["url"]
    result = {"source": name, "url": url, "ok": False, "items": [], "error": None}

    try:
        response = session.get(url, timeout=fetch_cfg["timeout_seconds"])
    except Exception as exc:  # noqa: BLE001 - 抓取失败的种类太多，统一报告
        result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        return result

    if response.status_code != 200:
        result["error"] = f"HTTP {response.status_code}"
        return result

    content_type = (response.headers.get("content-type") or "").lower()
    if "html" in content_type:
        # 这是最重要的一条检查。RSS 源返回 HTML 通常意味着：
        # 站点改版了，订阅地址 302 到了首页。它不会报错，只会静默返回一条都没有。
        result["error"] = (
            f"返回的是 HTML 而不是 XML（content-type: {content_type[:40]}）。\n"
            f"      通常意味着这个订阅地址已经失效、被重定向到首页了。"
        )
        return result

    try:
        items = parse_feed(response.content, name)
    except ValueError as exc:
        result["error"] = str(exc)
        return result

    if not items:
        result["error"] = "抓到内容但解析不出任何条目，源可能改了格式。"
        return result

    if limit:
        items = items[:limit]
    elif fetch_cfg.get("max_items_per_source"):
        items = items[: fetch_cfg["max_items_per_source"]]

    result["ok"] = True
    result["items"] = items
    return result


def missing_feed_hosts(sources: list[dict], policy: dict) -> list[str]:
    """找出不在 feed_hosts 白名单里的信息源域名。

    和模型层用的是同一个原则：**默认拒绝，而不是默认允许**。

    你可能会觉得这是多余的官僚步骤 —— 源本来就是你自己在 sources.json 里写的，
    为什么还要在 policy.json 里再确认一遍？

    因为这两件事的后果不同：
      模型调用是【把数据发出去】，信息源是【把数据拿回来】。
      而更要紧的是，白名单保护的从来不是"你主动做的事"，
      而是"某个库或某次改动顺手做的事"。写两处的成本是几秒钟，
      换来的是"每一次出网都是我按下的"这个性质。
    """
    from urllib.parse import urlparse

    allowed = set((policy.get("network") or {}).get("feed_hosts") or [])
    missing: list[str] = []
    for source in sources:
        host = urlparse(source.get("url", "")).hostname or ""
        if host and host not in allowed and host not in missing:
            missing.append(host)
    return missing


def collect(
    *,
    only_source: str | None = None,
    limit: int = 0,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    import requests

    config = load_sources()
    fetch_cfg = {**DEFAULT_FETCH, **(config.get("fetch") or {})}
    sources = [s for s in config.get("sources", []) if s.get("enabled") and s.get("url")]

    if only_source:
        sources = [s for s in sources if s.get("name") == only_source]
        if not sources:
            return {"fatal": f"没有名为 {only_source!r} 的已启用源。", "results": []}

    if not sources:
        return {
            "fatal": "config/sources.json 里没有启用任何源。\n"
                     "  把你要盯的源填上 url 并把 enabled 改成 true。",
            "results": [],
        }

    policy = load_policy()
    missing = missing_feed_hosts(sources, policy)
    if missing:
        listing = ",\n      ".join(f'"{host}"' for host in missing)
        return {
            "fatal": (
                "以下信息源域名不在白名单里，已拒绝抓取：\n"
                + "\n".join(f"      {host}" for host in missing)
                + "\n\n  打开 config/policy.json，把 network.feed_hosts 改成：\n"
                + "    \"feed_hosts\": [\n      " + listing + "\n    ]\n\n"
                + "  这一步故意设计成手动的 —— 出网应该是有意识的决定。"
            ),
            "results": [],
        }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    state = load_json_file(STATE_PATH, {"sources": {}})
    state.setdefault("sources", {})

    session = requests.Session()
    session.headers.update({"User-Agent": fetch_cfg["user_agent"]})

    results: list[dict] = []
    stats = {"new": 0, "dup_url": 0, "dup_content": 0, "too_short": 0, "stored": 0}

    for position, source in enumerate(sources, 1):
        name = source.get("name") or source["url"]
        if verbose:
            print(f"\n[{position}/{len(sources)}] {name}")

        result = fetch_source(source, fetch_cfg, limit, session)
        results.append(result)

        previous = state["sources"].get(name, {})
        state["sources"][name] = {
            "url": source["url"],
            "last_run": now_iso(),
            "ok": result["ok"],
            "items_found": len(result["items"]),
            "error": result["error"],
            # 连续失败次数是"大声失败"的基础：一次失败可能是抖动，连着三次就是坏了
            "consecutive_failures": 0 if result["ok"] else int(previous.get("consecutive_failures", 0)) + 1,
        }

        if not result["ok"]:
            if verbose:
                print(f"      失败：{result['error']}")
            if position < len(sources):
                time.sleep(fetch_cfg["delay_between_sources"])
            continue

        if verbose:
            print(f"      抓到 {len(result['items'])} 条")

        for item in result["items"]:
            item_id = make_id(item["url"]) if item["url"] else content_hash(item["title"], item["body"])
            digest = content_hash(item["title"], item["body"])

            if item_id in index["items"] and not force:
                stats["dup_url"] += 1
                continue
            existing_by_content = index["content_hashes"].get(digest)
            if existing_by_content and existing_by_content != item_id and not force:
                stats["dup_content"] += 1
                continue

            body = item["body"][: fetch_cfg["max_body_chars"]]
            if len(body) < fetch_cfg["min_body_chars"]:
                # 很多 RSS 只给一句话摘要。用它做摘要评估毫无意义 ——
                # 必须在这里挡掉，否则你会用一批"没有原文"的样本去测"摘要准不准"。
                stats["too_short"] += 1
                continue

            record = {
                "id": item_id,
                "title": item["title"],
                "url": item["url"],
                "source": item["source"],
                "published": item["published"],
                "fetched_at": now_iso(),
                "body": body,
                "body_chars": len(body),
                "truncated": len(item["body"]) > len(body),
                "content_hash": digest,
                "processed": False,
            }
            atomic_write_json(RAW_DIR / f"{item_id}.json", record)

            index["items"][item_id] = {
                "url": item["url"],
                "source": item["source"],
                "title": item["title"],
                "content_hash": digest,
                "body_chars": len(body),
                "first_seen": record["fetched_at"],
                "processed": False,
            }
            index["content_hashes"][digest] = item_id
            stats["stored"] += 1
            if verbose:
                mark = " [截断]" if record["truncated"] else ""
                print(f"        + {item['title'][:44]}{mark}  ({len(body)} 字)")

        if position < len(sources):
            time.sleep(fetch_cfg["delay_between_sources"])

    atomic_write_json(INDEX_PATH, index)
    atomic_write_json(STATE_PATH, state)

    ok_sources = [r for r in results if r["ok"]]
    failed_sources = [r for r in results if not r["ok"]]

    return {
        "fatal": None,
        "results": results,
        "stats": stats,
        "ok_count": len(ok_sources),
        "failed_count": len(failed_sources),
        "total_count": len(results),
        "index_size": len(index["items"]),
        "fetch_cfg": fetch_cfg,
    }


# ------------------------------------------------------------------ 本地投喂


INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
INBOX_DONE_NOTE = "data/inbox/_done.json"
INBOX_SOURCE_NAME = "本地投喂"
TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
HTML_SUFFIXES = {".html", ".htm"}


def _split_title(text: str, fallback: str) -> tuple[str, str]:
    """从纯文本里猜标题。

    规则（按可靠性从高到低）：

        1. `#` 开头的行                    —— 明确的标题标记，信它
        2. 短、且不以句号结尾的第一行       —— 像标题，信它
        3. 其他情况                        —— 用文件名，正文保留全文

    【第 2 条为什么还要看标点】

    只看长度是不够的：一个 48 字的完整句子完全在"短"的范围里，
    但它显然是个段落而不是标题。

    加一条"标题几乎不会以句号结尾"之后，误判率立刻降下来。
    这个思路可以推广：**当一个判据只能抓住一半特征时，
    与其把阈值调来调去，不如换一个正交的特征。**
    长度是量的特征，结尾标点是质的特征，两者一起用比只调长度有效得多。

    【第 3 条为什么存在】

    .txt 文档如果只有一段很长的话，前两条都不成立。
    这时的正确答案不是硬猜，而是退回文件名 ——
    一个"验证乙"这样的标题，比一句被截断的长句对人有用的多。
    **猜不出来的时候，退回到一个诚实的近似，比编一个精确的错答案强。**
    """
    lines = text.splitlines()
    sentence_endings = ("。", "！", "？", ".", "!", "?", "；", ";")

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            body = "\n".join(lines[index + 1 :]).strip()
            return (title or fallback), (body or text.strip())

        remainder = "\n".join(lines[index + 1 :]).strip()
        looks_like_title = (
            len(stripped) <= 60
            and bool(remainder)
            and not stripped.endswith(sentence_endings)
        )
        if looks_like_title:
            return stripped, remainder

        return fallback, text.strip()
    return fallback, text.strip()


def read_inbox_file(path: Path) -> tuple[str, str] | None:
    suffix = path.suffix.lower()
    try:
        if suffix in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
            if not text:
                return None
            return _split_title(text, path.stem)

        if suffix in HTML_SUFFIXES:
            raw = path.read_text(encoding="utf-8-sig", errors="replace")
            match = re.search(r"(?is)<title[^>]*>(.*?)</title>", raw)
            title = strip_html(match.group(1)) if match else path.stem
            return (title or path.stem), strip_html(raw)

        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            body = str(payload.get("body") or payload.get("text") or payload.get("content") or "")
            title = str(payload.get("title") or path.stem)
            body = body.strip()
            return (title, body) if body else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def import_inbox(verbose: bool = True) -> dict:
    """把 data/inbox/ 里的文档当作「来源 = 本地投喂」导入 data/raw/。

    【为什么需要这个功能】

    实测结论（用你自己的网络跑出来的，不是推测）：
      国内 RSS 源（量子位、InfoQ）只给 7-68 字的一句话摘要，【没有正文】；
      外文源（Google、DeepMind、HuggingFace）在你的网络下间歇性 SSL 失败。

    所以"从 RSS 抓全文来做摘要评估"这个方案在当前环境下不成立 ——
    抓回来的东西里根本没有可以摘要的正文。

    【修法不是换源，是解耦】

    评估样本的来源，和生产线发现线索的来源，本该是两件事：

      生产线（信息雷达）：RSS 负责【发现】——标题、链接、一句话摘要。这就够了，
                          它只需要帮你决定"这篇要不要看"。
      评估集：需要的是【你有全文的、你真正关心的】文档。它从哪来不重要 ——
              RSS 全文、网页另存、PDF 转文本、微信公众号复制粘贴，都行。

    把它们绑在一起，是这个项目里第一个真实的架构错误。
    它值得被记住：**数据层的形态决定了评估能不能做，而换模型救不回来。**

    用法：把文档丢进 data/inbox/，然后 run.cmd inbox。
    支持 .md / .txt / .html / .json（json 需要含 title 与 body 字段）。
    以 _ 开头的文件会被忽略（方便放说明文件）。
    """
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    index = load_index()
    done_state = load_json_file(PROJECT_ROOT / INBOX_DONE_NOTE, {"imported": {}})
    done_state.setdefault("imported", {})

    candidates = [
        path
        for path in sorted(INBOX_DIR.iterdir())
        if path.is_file()
        and not path.name.startswith("_")
        and path.suffix.lower() in (TEXT_SUFFIXES | HTML_SUFFIXES | {".json"})
    ]

    if not candidates:
        return {
            "empty": True,
            "inbox": str(INBOX_DIR),
            "imported": 0,
            "skipped": 0,
            "too_short": 0,
            "files": [],
        }

    fetch_cfg = {**DEFAULT_FETCH, **(load_sources().get("fetch") or {})}
    imported = skipped = too_short = 0
    entries: list[dict] = []

    for path in candidates:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        parsed = read_inbox_file(path)
        if not parsed:
            skipped += 1
            if verbose:
                print(f"  跳过（读不出内容）：{relative}")
            continue

        title, body = parsed
        if len(body) < fetch_cfg["min_body_chars"]:
            too_short += 1
            if verbose:
                print(f"  跳过（正文仅 {len(body)} 字，少于 {fetch_cfg['min_body_chars']}）：{relative}")
            continue

        item_id = make_id(f"inbox:{relative}")
        digest = content_hash(title, body)

        if item_id in index["items"]:
            previous = index["items"][item_id]
            if previous.get("content_hash") == digest:
                skipped += 1
                continue

        body = body[: fetch_cfg["max_body_chars"]]
        record = {
            "id": item_id,
            "title": title,
            "url": "",
            "source": INBOX_SOURCE_NAME,
            "origin_file": relative,
            "published": None,
            "fetched_at": now_iso(),
            "body": body,
            "body_chars": len(body),
            "truncated": False,
            "content_hash": digest,
            "processed": False,
        }
        atomic_write_json(RAW_DIR / f"{item_id}.json", record)

        index["items"][item_id] = {
            "url": "",
            "source": INBOX_SOURCE_NAME,
            "title": title,
            "origin_file": relative,
            "content_hash": digest,
            "body_chars": len(body),
            "first_seen": record["fetched_at"],
            "processed": False,
        }
        index["content_hashes"][digest] = item_id
        done_state["imported"][relative] = record["fetched_at"]
        imported += 1
        entries.append({"file": relative, "title": title, "chars": len(body)})
        if verbose:
            print(f"  + {title[:44]}  ({len(body)} 字，来自 {path.name})")

    atomic_write_json(INDEX_PATH, index)
    atomic_write_json(PROJECT_ROOT / INBOX_DONE_NOTE, done_state)

    return {
        "empty": False,
        "inbox": str(INBOX_DIR),
        "imported": imported,
        "skipped": skipped,
        "too_short": too_short,
        "files": entries,
        "index_size": len(index["items"]),
    }


# ------------------------------------------------------------------ 入口


def print_report(outcome: dict) -> None:
    if outcome.get("fatal"):
        print(f"\n[错误] {outcome['fatal']}\n")
        return

    stats = outcome["stats"]
    cfg = outcome["fetch_cfg"]
    print()
    print("=" * 68)
    print("  抓取结果")
    print("=" * 68)
    print(f"  源           成功 {outcome['ok_count']} / {outcome['total_count']}")
    print(f"  新增入库     {stats['stored']} 条")
    print(f"  重复跳过     URL {stats['dup_url']} 条，内容指纹 {stats['dup_content']} 条")
    print(f"  正文过短丢弃 {stats['too_short']} 条（少于 {cfg['min_body_chars']} 字）")
    print(f"  索引总量     {outcome['index_size']} 条")

    failed = [r for r in outcome["results"] if not r["ok"]]
    if failed:
        print()
        print("  " + "-" * 64)
        print("  失败的源：")
        for item in failed:
            print(f"    [!!] {item['source']}")
            print(f"         {item['error']}")
        print()
        print("  这些源不会自己恢复。连续失败会被记进 data/cache/sources-state.json，")
        print("  跑 run.cmd status 就能看到。")
        print("  【不要忽略这一段】—— 一个悄悄死掉的源会让你以为今天没有新闻。")

    if stats["stored"] == 0 and outcome["ok_count"] > 0:
        print()
        print("  没有新条目。（可能是源没有更新，也可能都已经抓过了。）")

    print()
    print("  下一步：run.cmd sample -N 20  从库里抽 20 条，生成标注表")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取并归一化信息源")
    parser.add_argument("--source", "-Source", default=None, help="只抓指定名字的源")
    parser.add_argument("--limit", "-Limit", "-limit", type=int, default=0, help="每个源最多取 N 条")
    parser.add_argument("--force", action="store_true", help="忽略去重，强制重新入库")
    parser.add_argument("--list", action="store_true", help="只列出配置里的源，不联网")
    parser.add_argument("--inbox", action="store_true", help="导入 data/inbox/ 里的本地文档")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.inbox:
        print()
        print(f"  导入本地投喂：{INBOX_DIR}")
        print()
        result = import_inbox(verbose=not args.quiet)
        print()
        print("=" * 68)
        if result.get("empty"):
            print("  data/inbox/ 里没有可导入的文档。")
            print()
            print("  把你想让系统处理的文档丢进去就行：")
            print("    .md / .txt     —— 直接放，第一行或 # 开头的行会当标题")
            print("    .html          —— 网页另存为，会自动去标签")
            print("    .json          —— 需要含 title 和 body 两个字段")
            print()
            print("  【评估集应该用这些文档】，而不是用 RSS 抓来的东西 ——")
            print("  国内 RSS 只给一句话摘要，没有正文可以摘要。")
        else:
            print("  导入结果")
            print("=" * 68)
            print(f"  新增入库   {result['imported']} 篇")
            print(f"  重复跳过   {result['skipped']} 篇")
            print(f"  正文过短   {result['too_short']} 篇")
            print(f"  索引总量   {result.get('index_size', 0)} 条")
            if result["imported"]:
                print()
                print("  下一步：run.cmd sample -N 20")
        print()
        return 0

    if args.list:
        config = load_sources()
        print()
        print("  config/sources.json 里的源：")
        print()
        for source in config.get("sources", []):
            flag = "启用" if source.get("enabled") else "停用"
            url = source.get("url") or "(未填地址)"
            print(f"    [{flag}] {source.get('name', '(无名)')}")
            print(f"           {url}")
            if source.get("note"):
                print(f"           备注：{source['note']}")
        print()
        return 0

    outcome = collect(
        only_source=args.source,
        limit=args.limit,
        force=args.force,
        verbose=not args.quiet,
    )
    print_report(outcome)

    if outcome.get("fatal"):
        return 2
    # 全部源都失败 = 大声失败。退出码非零，这样以后接定时任务时它会被发现。
    if outcome["failed_count"] == outcome["total_count"] and outcome["total_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
