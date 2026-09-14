"""E7-A 自监督预训练（协议 v3）：同窗口双视图一致性（NT-Xent）

- 主干 KLineTransformer 一行不改：通过 forward hook 抓 norm 输出做表示
- 视图 = 收益率空间抖动（协议 §3.2 审计口径：ε~N(0, α·σ_window/√L)）
- 负样本排除：同品种且锚定日距离 ≤ guard_days 的样本对不得互为负样本
  （覆盖分钟窗口重叠 + 日K/外盘上下文重叠 + 时间近邻保护区）；
  排除后有效负样本数记录 effective_negative_count，不静默缩小分母
- 预训练头（2层 MLP → 128 维）预训练后丢弃，不进微调
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class PretrainViewDataset(Dataset):
    """在 KLineDataset（return_raw=True）外包一层：附加锚定日/品种序/归一化统计。"""

    def __init__(self, base: Dataset, sym_idx: int, anchor_days: np.ndarray):
        self.base = base
        self.sym_idx = sym_idx
        self.anchor_days = anchor_days.astype(np.int64)
        self.mean = torch.from_numpy(np.asarray(base.mean, dtype=np.float32))
        self.std = torch.from_numpy(np.asarray(base.std, dtype=np.float32))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        item = dict(self.base[i])
        item["sym_idx"] = torch.tensor(self.sym_idx, dtype=torch.long)
        item["anchor_day"] = torch.tensor(self.anchor_days[i], dtype=torch.long)
        item["norm_mean"] = self.mean
        item["norm_std"] = self.std
        return item


def return_space_jitter(seq: torch.Tensor, raw: torch.Tensor,
                        mean: torch.Tensor, std: torch.Tensor,
                        alpha: float) -> torch.Tensor:
    """协议 §3.2：对每根 bar 的 log 收益加 ε~N(0, α·σ_w/√L)，重建价格路径。

    seq: [B,S,C] 归一化特征；raw: [B,S,C] 原始 OHLC(+可选 volume/OI)；
    mean/std: [C] 全序列归一化统计。只动 OHLC 前 4 通道，volume/OI 不变。
    """
    closes = raw[:, :, 3].clamp(min=1e-8)
    rets = torch.log(closes[:, 1:] / closes[:, :-1])          # [B, L]
    sigma_w = rets.std(dim=1, keepdim=True).clamp(min=1e-8)   # [B,1]
    noise = torch.randn_like(rets) * (alpha * sigma_w / math.sqrt(rets.shape[1]))
    ratio = torch.exp(torch.cumsum(noise, dim=1))
    ratio = torch.cat([torch.ones_like(ratio[:, :1]), ratio], dim=1)  # [B,S]
    new_raw = raw.clone()
    n_price = min(4, raw.shape[2])
    new_raw[:, :, :n_price] = raw[:, :, :n_price] * ratio.unsqueeze(-1)
    return (new_raw - mean.view(1, 1, -1)) / std.view(1, 1, -1)


class ProjectionHead(nn.Module):
    """2 层 MLP → 128 维 → L2 归一化。预训练后丢弃。"""

    def __init__(self, hidden: int, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.mlp(x), dim=-1)


def build_allowed_mask(sym_idx: torch.Tensor, anchor_day: torch.Tensor,
                       guard_days: int) -> torch.Tensor:
    """[B,B] bool：True=允许互为负样本。同品种且锚定日距离 ≤ guard_days 排除。"""
    same_sym = sym_idx.view(-1, 1) == sym_idx.view(1, -1)
    near = (anchor_day.view(-1, 1) - anchor_day.view(1, -1)).abs() <= guard_days
    return ~(same_sym & near)


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, allowed: torch.Tensor,
            tau: float) -> tuple[torch.Tensor, float]:
    """带排除掩码的 NT-Xent。allowed: [B,B]。返回 (loss, 平均有效负样本数)。"""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)                       # [2B, H]
    sim = (z @ z.t()) / tau                              # [2B, 2B]
    # 全掩码：块内/块间都用同一 allowed（对称）；正样本位永远保留
    mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=z.device)
    mask[:B, B:] = allowed
    mask[B:, :B] = allowed
    mask[:B, :B] = allowed
    mask[B:, B:] = allowed
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    mask[torch.arange(2 * B), pos_idx] = True            # 正样本始终可见
    sim = sim.masked_fill(~mask, float("-inf"))
    sim.fill_diagonal_(float("-inf"))                    # 自身永远排除（覆盖正样本位误设）
    sim[torch.arange(2 * B), pos_idx] = (z * z[pos_idx]).sum(-1) / tau  # 恢复正样本 logit
    targets = pos_idx
    loss = F.cross_entropy(sim, targets)
    # 有效负样本数：每行允许的负样本（= 掩码 True 数 - 1 个正样本）
    eff_neg = (mask.sum(dim=1).float() - 1).mean().item()
    return loss, eff_neg


class PretrainTrainer:
    """E7-A 预训练循环：双视图 → 主干（hook 抓表示）→ 投影头 → 排除版 NT-Xent。"""

    def __init__(self, model, loader, lr: float = 3e-4, weight_decay: float = 0.01,
                 max_epochs: int = 20, tau: float = 0.1, alpha: float = 0.10,
                 guard_days: int = 30, device: str | None = None,
                 save_dir: str | Path = "./checkpoints_pretrain"):
        self.model = model
        self.loader = loader
        self.lr = lr
        self.max_epochs = max_epochs
        self.tau = tau
        self.alpha = alpha
        self.guard_days = guard_days
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        hidden = model.config.hidden_size
        self.proj = ProjectionHead(hidden).to(self.device)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.proj.parameters()),
            lr=lr, weight_decay=weight_decay)
        # 表示捕获：hook 主干最终 norm 输出，取 readout 位置（默认 last）
        self._repr: torch.Tensor | None = None
        self.model.norm.register_forward_hook(self._grab)

    def _grab(self, module, inputs, output):
        self._repr = output

    def _pooled(self, x_norm_out: torch.Tensor) -> torch.Tensor:
        if getattr(self.model, "readout", "last") == "cls":
            return x_norm_out[:, 0, :]
        return x_norm_out[:, -1, :]

    def _forward_repr(self, batch, jitter: bool) -> torch.Tensor:
        x = batch["seq"].to(self.device)
        if jitter:
            raw = batch["raw_seq"].to(self.device)
            mean = batch["norm_mean"][0].to(self.device)  # 同 dataset 统计一致
            std = batch["norm_std"][0].to(self.device)
            x = return_space_jitter(x, raw, mean, std, self.alpha)
        kw = {}
        for k in ("temporal_feat", "time_pos", "freq_feat", "symbol_id",
                  "daily_ctx", "foreign_ctx", "fine_ctx", "cross_ctx", "cross_mask"):
            v = batch.get(k)
            if v is not None:
                kw[k] = v.to(self.device)
        self.model(**{"kline_seq": x, **kw})
        rep = self._pooled(self._repr)
        self._repr = None
        return rep

    def fit(self):
        print(f"\n== E7-A 预训练 | α={self.alpha} | τ={self.tau} | "
              f"guard={self.guard_days}d | lr={self.lr} 固定 | epochs={self.max_epochs}")
        print("协议提醒: 首 epoch InfoNCE 若≈0，说明视图太容易，需上调 α 或加遮蔽 ==\n")
        losses = []
        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            ep_loss, ep_neg, n_step = 0.0, 0.0, 0
            for batch in self.loader:
                allowed = build_allowed_mask(
                    batch["sym_idx"].to(self.device),
                    batch["anchor_day"].to(self.device),
                    self.guard_days)
                r1 = self._forward_repr(batch, jitter=True)
                r2 = self._forward_repr(batch, jitter=True)
                z1, z2 = self.proj(r1), self.proj(r2)
                loss, eff_neg = nt_xent(z1, z2, allowed, self.tau)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.proj.parameters()), 1.0)
                self.optimizer.step()
                ep_loss += loss.item()
                ep_neg += eff_neg
                n_step += 1
            avg = ep_loss / max(n_step, 1)
            losses.append(avg)
            print(f"Epoch {epoch:3d}/{self.max_epochs} | InfoNCE={avg:.4f} | "
                  f"有效负样本={ep_neg / max(n_step, 1):.0f} | "
                  f"time={time.time() - t0:.1f}s")
            if epoch == 1 and avg < 0.5:
                print(f"  ⚠️ 首 epoch InfoNCE={avg:.4f} 偏低：视图可能太容易，"
                      f"按协议考虑上调 α 或加遮蔽")
            # 平台期即停：近 5 轮改善 < 0.005
            if len(losses) >= 6 and (max(losses[-6:-1]) - losses[-1]) < 0.005:
                print(f"⏹ 对比损失平台期，提前停（epoch {epoch}）")
                break
        out = self.save_dir / "pretrain_encoder.pt"
        torch.save({
            "model": self.model.state_dict(),
            "config": self.model.config,
            "e7_manifest": {
                "aug": f"E7-aug0 (return jitter α={self.alpha})",
                "tau": self.tau, "guard_days": self.guard_days,
                "lr": self.lr, "epochs_run": len(losses),
                "final_info_nce": losses[-1], "loss_curve": losses,
                "data_scope": "train split only（协议 v3 §6.1）",
            },
        }, out)
        print(f"\n已保存预训练 encoder: {out}")
        print("下一步：probe 门禁（label/path/symbol）通过后再微调；投影头不进微调。")
        return losses
