"""按【显示宽度】处理文本，而不是按字符数。

【为什么需要这个】

中文（以及日文、韩文）字符在终端里占 **2 列**，英文字符占 1 列。
而 Python 的 `len()` 和切片 `[:44]` 只数字符个数，不管宽度。

后果很具体。用 `title[:44]` 去排一个列表：

    英文标题截到 44 个字符  →  占 44 列
    中文标题截到 44 个字符  →  占 88 列（实测最宽 81 列）

同一列里，中文行溢出到下一行、英文行留一大片空白，
精心排的对齐全乱，而"溢出"看起来像是排版坏了。

**这类 bug 只用英文数据永远测不出来。** 而这个项目的数据一半是中文。

（这不是猜的：用真实抓回来的 122 条跑 `run.cmd pipeline --dry-run`，
 量出来标题列宽度从 16 列到 81 列不等，原因是截断按字符数算。）
"""

from __future__ import annotations

import unicodedata

ELLIPSIS = "…"


def char_width(ch: str) -> int:
    """单个字符占几列。东亚全角字符占 2 列，其余占 1 列。"""
    return 2 if unicodedata.east_asian_width(ch) in "WF" else 1


def width(text: str) -> int:
    """这段文字在终端里占几列。"""
    return sum(char_width(ch) for ch in text)


def truncate(text: str, columns: int, *, ellipsis: str = ELLIPSIS) -> str:
    """按显示宽度截断，超出的部分换成省略号。

    截断时补一个省略号，是为了让你能看出"这里被截了"——
    否则一个正好 44 列的标题和一个被截到 44 列的标题长得一样。
    """
    text = str(text)
    if columns <= 0:
        return ""
    if width(text) <= columns:
        return text

    budget = columns - width(ellipsis)
    if budget <= 0:
        return ellipsis[:columns]

    kept: list[str] = []
    used = 0
    for ch in text:
        w = char_width(ch)
        if used + w > budget:
            break
        kept.append(ch)
        used += w
    return "".join(kept) + ellipsis


def pad(text: str, columns: int) -> str:
    """按显示宽度右侧补空格，让多行能对齐。"""
    text = str(text)
    gap = columns - width(text)
    return text + " " * gap if gap > 0 else text
