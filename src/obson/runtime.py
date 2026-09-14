# -*- coding: utf-8 -*-
"""生产运行时口径解析 —— 防止 theta_q 等训练超参靠命令行记忆漂移（老师审查 #6）

约定：训练脚本把 theta_q / soft_label 写进 ckpt config；消费脚本（信号/回测/EV表）
一律从 ckpt 恢复。命令行 --theta-q 只作显式覆盖，且与 ckpt 不一致时默认报错。
"""
from __future__ import annotations


def resolve_theta_q(cli_value, cfg, fallback: float = 0.90,
                    force: bool = False, script: str = "") -> tuple[float, str]:
    """返回 (theta_q, 来源说明)。
    - cli None：ckpt 有则用 ckpt，否则回退 fallback（老 ckpt 无此字段；
      生产冠军 ckpt 训练口径即 0.90，回退值与之相等，回归不受影响）
    - cli 显式传入且与 ckpt 不一致：报错退出，除非 force=True
    """
    ckpt_v = getattr(cfg, "theta_q", None)
    if cli_value is None:
        if ckpt_v is not None:
            return float(ckpt_v), "ckpt"
        return fallback, f"回退默认{fallback}（老ckpt无此字段）"
    if ckpt_v is not None and abs(float(cli_value) - float(ckpt_v)) > 1e-9 and not force:
        raise SystemExit(
            f"[{script}] --theta-q {cli_value} 与 ckpt 训练口径 {ckpt_v} 不一致；"
            f"确认要覆盖请加 --force-theta"
        )
    return float(cli_value), "命令行" + ("（强制覆盖）" if force and ckpt_v is not None
                                       and abs(float(cli_value) - float(ckpt_v or 0)) > 1e-9 else "")
