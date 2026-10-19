"""把 info-radar 推到 GitHub。

为什么做成脚本而不是让你手打命令：

  因为要判断的事情有四五件 —— 有没有登录、仓库建过没有、
  远程有没有配、分支叫 master 还是 main。
  任何一步判断错，你会得到一句英文报错，然后不知道接下来该干什么。

所以这里把判断都做完，只把【必须你授权的那一下】留给你。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "info-radar"


def run(cmd: list[str], *, check: bool = False) -> tuple[int, str]:
    """跑一条命令，返回 (退出码, 输出)。"""
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180,
        )
        return proc.returncode, (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        return 127, "找不到命令"
    except subprocess.TimeoutExpired:
        return 124, "超时"


def hr(char: str = "=") -> str:
    return char * 66


def main() -> int:
    print()
    print(hr())
    print("  推送到 GitHub")
    print(hr())

    # ---------------- 1. 本地仓库检查
    code, out = run(["git", "rev-parse", "--is-inside-work-tree"])
    if code != 0:
        print("\n  这不是一个 git 仓库。\n")
        return 1

    code, branch = run(["git", "branch", "--show-current"])
    branch = branch.strip() or "master"
    code, count = run(["git", "rev-list", "--count", "HEAD"])
    commits = count.strip() or "?"

    print(f"\n  本地：{commits} 个提交，分支 {branch}")

    code, status = run(["git", "status", "--porcelain"])
    if status.strip():
        print("\n  ! 还有没提交的改动：")
        for line in status.strip().splitlines()[:10]:
            print(f"      {line}")
        print("\n  先把这些提交掉再推，否则推上去的不是完整的东西。\n")
        return 1
    print("  工作区干净")

    # ---------------- 2. 有没有登录
    code, auth = run(["gh", "auth", "status"])
    if code != 0:
        print()
        print(hr("-"))
        print("  还差你授权一次")
        print(hr("-"))
        print()
        print("  GitHub 需要你本人登录一次，这一步我替你做不了")
        print("  （否则任何人都能把东西推到你的账号下）。")
        print()
        print("  做完这一件事，再双击一次这个文件就行：")
        print()
        print("      gh auth login")
        print()
        print("  它会问你几个问题，一路回车选默认即可：")
        print("     · What account?          → GitHub.com")
        print("     · Preferred protocol?    → HTTPS")
        print("     · Authenticate Git?      → Yes")
        print("     · How to login?          → Login with a web browser")
        print()
        print("  然后把浏览器里显示的一串一次性代码填进去。")
        print()
        print("  如果不想用命令行：也可以自己去 github.com/new 建一个空仓库，")
        print("  把网址发给我，我用你电脑里已有的凭据推上去。")
        print()
        return 1
    print("  GitHub 已登录")

    # ---------------- 3. 仓库建过没有
    code, who = run(["gh", "api", "user", "--jq", ".login"])
    owner = who.strip()
    if not owner:
        print("\n  ! 拿不到 GitHub 用户名，登录状态可能失效了。\n")
        return 1
    print(f"  账号：{owner}")

    repo = DEFAULT_REPO
    code, existing = run(["gh", "repo", "view", f"{owner}/{repo}", "--json", "nameWithOwner"])
    repo_exists = code == 0

    # ---------------- 4. 配远程
    code, remotes = run(["git", "remote", "-v"])
    has_origin = "origin" in remotes

    if repo_exists:
        print(f"  仓库已存在：https://github.com/{owner}/{repo}")
        if not has_origin:
            code, out = run([
                "git", "remote", "add", "origin",
                f"https://github.com/{owner}/{repo}.git",
            ])
            print("  已配置远程 origin")
    else:
        if has_origin and f"{owner}/{repo}" in remotes:
            # 远程已配但仓库不存在 —— 用 push 时 gh 会建
            pass
        print(f"\n  仓库还不存在，准备创建：{owner}/{repo}（私有）")
        print("  —— 私有仓库的意思是：只有你能看到。想公开随时可以改。")

    # ---------------- 5. 推
    print()
    print(hr("-"))
    print("  开始推送")
    print(hr("-"))
    print()

    if repo_exists:
        code, out = run(["git", "push", "-u", "origin", branch])
    elif has_origin:
        code, out = run(["git", "push", "-u", "origin", branch])
    else:
        code, out = run([
            "gh", "repo", "create", repo, "--private", "--source=.",
            "--remote=origin", "--push",
        ])

    for line in out.strip().splitlines():
        print(f"  {line}")

    print()
    if code == 0:
        print(hr())
        print("  推上去了")
        print(hr())
        print()
        print(f"  地址：https://github.com/{owner}/{repo}")
        print()
        print("  以后改了东西，只要两步：")
        print("      git add -A")
        print('      git commit -m "改了什么"')
        print("      git push")
        print()
        print("  （或者再双击一次这个文件，它会自动推到已有的仓库）")
        print()
        return 0

    print(hr())
    print("  推送失败了")
    print(hr())
    print()
    print("  上面的报错如果提到 authentication / credential，说明登录又失效了，")
    print("  重新跑一次 gh auth login 就行。")
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
