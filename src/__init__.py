"""info-radar：个人 AI 工作系统。

六层结构与它在这个包里的对应关系：

    ⓞ 任务      config/task.json
    ⓪ 评估      src/evaluate.py, eval/
    ① 数据      src/collect.py            (S2)
    ② 模型      src/llm.py
    ③ 工作流    src/pipeline.py           (S3)
    ④ 分发      src/archive.py            (S4)
    ⑤ 权限审计  src/secrets.py, src/ledger.py, config/policy.json
"""

__all__ = ["secrets", "ledger", "config", "llm", "task", "collect", "pipeline", "evaluate"]
__version__ = "0.3.0"
