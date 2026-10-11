"""第 ⑤ / ② 层 · 密钥与脱敏

这一层只有三条规则，其余都是它们的实现：

  规则 1  密钥只从项目根目录的 .env 读取。
          不写进代码、不写进 config/、不进 git、不进聊天记录。

  规则 2  任何要被写下来的文本（日志、账本、异常信息）都必须先过 mask()。
          账本里只允许出现密钥指纹（末 4 位）。

  规则 3  缺密钥必须【大声失败】，并且明确说出缺哪个变量名。
          静默返回空串会让服务端回一个 401，而你看不懂发生了什么 ——
          这是最低级的浪费时间的方式。

为什么不用 python-dotenv：这个解析器只有二十行，你需要能读懂它每一步。
能坏的地方越少越好。
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# 已加载的密钥明文。只存在于内存里，唯一用途是脱敏。
_LOADED_SECRETS: list[str] = []
_ENV_CACHE: dict[str, str] | None = None

# 变量名里带这些词的，才当密钥登记并脱敏。
_SECRET_NAME_RE = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.IGNORECASE)


class MissingKey(RuntimeError):
    """缺少必需的密钥。错误信息必须直接告诉人下一步做什么。"""


def _register(name: str, value: str) -> None:
    """登记一个需要脱敏的值。只有看起来像密钥的变量名才登记。"""
    if value and _SECRET_NAME_RE.search(name) and value not in _LOADED_SECRETS:
        _LOADED_SECRETS.append(value)


def _parse_env_file(path: Path) -> dict[str, str]:
    """极简 .env 解析：KEY=VALUE，忽略空行与 # 注释，去掉成对引号与行内注释。"""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].strip()
        if key:
            out[key] = value
    return out


def load_env(path: Path | None = None, *, refresh: bool = False) -> dict[str, str]:
    """读取 .env（带缓存）。

    约定：真实环境变量优先，.env 兜底。
    这样临时覆盖或 CI 注入不会被一个残留的 .env 顶掉。
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None and not refresh:
        return _ENV_CACHE

    from_file = _parse_env_file(path or ENV_PATH)
    for name, value in from_file.items():
        _register(name, value)
    # 真实环境变量只覆盖 .env 里已出现的键，不额外引入新键（新键由 get_key 直接查 os.environ）
    merged = dict(from_file)
    for name in from_file:
        if os.environ.get(name):
            merged[name] = os.environ[name]
            _register(name, merged[name])

    _ENV_CACHE = merged
    return merged


# 已经提醒过的变量名。同类提醒只说一次 —— 一个每次调用都刷屏的警告，
# 很快就会变成背景噪音，然后被无视。
_WARNED_KEYS: set[str] = set()


def key_problems(value: str) -> list[str]:
    """这个值像不像一个密钥？返回问题列表（空列表 = 看起来正常）。

    单独拆成纯函数，是为了它能被测试。
    "填错密钥"是一条会让人反复重新申请、却永远修不好的路径 ——
    值得有测试守着，而不是靠读代码相信它。
    """
    problems: list[str] = []
    if not value:
        return ["是空的"]
    if any(ch.isspace() for ch in value):
        problems.append("里面含空格或换行（多半是复制时多带了一段）")
    if len(value) < 16:
        problems.append(f"只有 {len(value)} 个字符（正常的密钥一般更长，可能没复制全）")
    if ":" in value or "：" in value:
        problems.append("里面含冒号（多半把「API Key:」这类文字也复制进来了）")
    if value.lower().startswith("bearer"):
        problems.append("以 bearer 开头（那是请求头里的写法，不是密钥本身）")
    return problems


def _sanity_check(name: str, value: str) -> None:
    """在密钥进入系统的那一刻提醒一次。

    最常见的两种填错方式都不会在填写时报错，只会在调用时表现为「密钥被拒绝」：

        1. 复制时多带了一段（"API Key: sk-xxx"、"Bearer sk-xxx"、前后空格）
        2. 只复制了一半

    这时用户会以为是【自己申请的方式不对】，于是回去重新申请 ——
    而真正的问题是复制多带了几个字。
    **让错误指向错误的方向，比不报错更浪费时间。**

    只是提醒，不阻止运行 —— 万一你的密钥真的长这样呢。
    """
    if name in _WARNED_KEYS:
        return
    problems = key_problems(value)
    if not problems:
        return
    _WARNED_KEYS.add(name)
    print(f"\n[提醒] {name} 看起来不太对：", file=sys.stderr)
    for problem in problems:
        print(f"         - {problem}", file=sys.stderr)
    print(f"       它现在的指纹是 {fingerprint(value)}。", file=sys.stderr)
    print("       如果下一步报「密钥被拒绝」，先回到 .env 检查这一行。\n", file=sys.stderr)


def get_key(env_name: str, *, required: bool = True) -> str:
    """取一个密钥。缺了就抛 MissingKey，并说清楚怎么补。"""
    value = (os.environ.get(env_name) or load_env().get(env_name) or "").strip()
    if not value:
        if required:
            raise MissingKey(
                f"缺少密钥：{env_name}\n"
                f"  1) 打开或新建文件：{ENV_PATH}\n"
                f"  2) 写入一行：{env_name}=你的密钥\n"
                f"  3) 保存后重新运行。\n"
                f"提醒：.env 不要提交到 git，也不要贴进任何聊天窗口。"
            )
        return ""
    _register(env_name, value)
    _sanity_check(env_name, value)
    return value


def fingerprint(secret: str) -> str:
    """密钥指纹：只留末 4 位，够用来核对『我填的是哪一把』，不够用来盗用。"""
    if not secret:
        return "(空)"
    return f"***{secret[-4:] if len(secret) >= 4 else secret}"


def mask(text: str) -> str:
    """把已知密钥从任意文本里抹掉。

    规则：任何写进文件或打印出来的东西，都要先过这一道。
    它是这套系统里编号最小的一个安全网，成本几乎为零。
    """
    if not text:
        return text
    for secret in _LOADED_SECRETS:
        if secret and len(secret) >= 8:
            text = text.replace(secret, fingerprint(secret))
    return text


def loaded_secret_count() -> int:
    """已登记待脱敏的密钥数量。自检用，不泄露内容。"""
    return len(_LOADED_SECRETS)
