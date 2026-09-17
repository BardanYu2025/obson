"""E12 形态骨干：per-bar Transformer（因果/双向双模式）+ 稠密监督头

token = 一根 bar。输出每根 bar 的上下文表示 h_t（[B,T,H]），
h_t 即"bar embedding"，供一切下游任务复用（冻结骨干 + 小头）。

模式纪律：
- causal=True  → 因果掩码，实盘下游唯一合法接入；
- causal=False → 全注意力，仅离线分析/probe；存档命名强制带 _causal/_bidir。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

N_SCALES = 3


@dataclass
class PatternConfig:
    feature_dim: int = 8
    hidden_size: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_size: int = 256
    dropout: float = 0.1
    max_len: int = 512
    n_symbols: int = 16
    n_freqs: int = 8
    causal: bool = True
    mask_ratio: float = 0.15   # 掩码重构比例


class PatternEncoder(nn.Module):
    def __init__(self, cfg: PatternConfig):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_size
        self.in_proj = nn.Linear(cfg.feature_dim, H)
        self.mask_token = nn.Parameter(torch.randn(1, 1, cfg.feature_dim) * 0.02)
        self.pos_emb = nn.Embedding(cfg.max_len, H)
        self.symbol_emb = nn.Embedding(cfg.n_symbols, H)
        self.freq_emb = nn.Embedding(cfg.n_freqs, H)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=cfg.num_heads, dim_feedforward=cfg.ff_size,
            dropout=cfg.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.num_layers)
        self.norm = nn.LayerNorm(H)
        # ── 稠密监督头（全部 per-bar）──
        self.seg_dir_head = nn.Linear(H, N_SCALES * 3)   # [B,T,3尺度,3类]
        self.time_head = nn.Linear(H, N_SCALES)          # bars_since_piv
        self.amp_head = nn.Linear(H, N_SCALES)           # amp_since_piv
        self.piv_head = nn.Linear(H, N_SCALES * 3)       # 0无/1高/2低
        self.quality_head = nn.Linear(H, N_SCALES)       # piv_amp（仅枢轴 bar 计损）
        self.recon_head = nn.Linear(H, 4)                # 掩码重构 OHLC

    def forward(self, x, symbol_id, freq_id, mask: torch.Tensor | None = None):
        """x [B,T,F]；mask [B,T] bool（True=该 bar 被遮蔽，需重构）。"""
        B, T, _ = x.shape
        if mask is not None:
            x = torch.where(mask.unsqueeze(-1), self.mask_token.expand(B, T, -1), x)
        h = self.in_proj(x)
        pos = torch.arange(T, device=x.device)
        h = h + self.pos_emb(pos).unsqueeze(0)
        h = h + self.symbol_emb(symbol_id).unsqueeze(1) + self.freq_emb(freq_id).unsqueeze(1)
        attn_mask = None
        if self.cfg.causal:
            attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h = self.norm(self.encoder(h, mask=attn_mask))     # [B,T,H]
        return {
            "h": h,
            "seg_dir": self.seg_dir_head(h).view(B, T, N_SCALES, 3),
            "bars_since": self.time_head(h),
            "amp_since": self.amp_head(h),
            "piv_cls": self.piv_head(h).view(B, T, N_SCALES, 3),
            "piv_amp": self.quality_head(h),
            "recon": self.recon_head(h),
        }


def pattern_losses(out: dict, batch: dict, mask: torch.Tensor | None,
                   piv_weight: float = 5.0) -> dict[str, torch.Tensor]:
    """分量记账（教师纪律：不许只看加权和）。返回 {name: loss}。"""
    device = out["h"].device
    seg_dir = batch["seg_dir"].to(device)          # [B,T,3]
    bars_since = batch["bars_since"].to(device)
    amp_since = batch["amp_since"].to(device)
    piv_cls = batch["piv_cls"].to(device)
    piv_amp = batch["piv_amp"].to(device)

    l_dir = nn.functional.cross_entropy(
        out["seg_dir"].reshape(-1, 3), seg_dir.reshape(-1))
    # 枢轴检测：少数类加权（枢轴占比 ~5%/尺度）
    cw = torch.tensor([1.0, piv_weight, piv_weight], device=device)
    l_piv = nn.functional.cross_entropy(
        out["piv_cls"].reshape(-1, 3), piv_cls.reshape(-1), weight=cw)
    # 坐标回归：log1p 压缩时间坐标的长尾
    l_time = nn.functional.smooth_l1_loss(
        torch.log1p(out["bars_since"]), torch.log1p(bars_since))
    l_amp = nn.functional.smooth_l1_loss(out["amp_since"], amp_since)
    # 成色：只在真枢轴 bar 上计损
    is_piv = piv_cls > 0
    if is_piv.any():
        l_qual = nn.functional.smooth_l1_loss(
            out["piv_amp"][is_piv], piv_amp[is_piv])
    else:
        l_qual = torch.tensor(0.0, device=device)
    losses = {"dir": l_dir, "piv": l_piv, "time": l_time, "amp": l_amp, "qual": l_qual}
    if mask is not None and mask.any():
        recon_t = batch["recon"].to(device)
        losses["recon"] = nn.functional.mse_loss(out["recon"][mask], recon_t[mask])
    return losses


@torch.no_grad()
def gate_metrics(out: dict, batch: dict, tol: int = 3) -> dict[str, float]:
    """v1.1 双轨门禁指标：
    - 理解轨（双向版用）：U1 枢轴 F1（精确位置）/ U2 段方向 BA / U3 坐标 R²
    - 前瞻轨（因果版用）：A1 容差事件 F1（预测落在真枢轴 ±tol bar 内算命中）、
      A2 段方向 BA 与坐标 R² 只在 confirmed 掩码（当时可知）样本上评估
    返回两种口径全套指标，由调用方按模型模式取用。
    """
    piv = out["piv_cls"].argmax(-1).cpu()            # [B,T,3]
    gt = batch["piv_cls"]
    conf = batch["confirmed"]                        # [B,T,3] bool
    f1s, f1t = [], []
    for s in range(N_SCALES):
        for c in (1, 2):
            p, g = (piv[..., s] == c), (gt[..., s] == c)
            # 精确口径（理解轨）
            tp = (p & g).sum().item()
            fp = (p & ~g).sum().item()
            fn = (~p & g).sum().item()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
            # 容差事件口径（前瞻轨）：真枢轴 ±tol 内有同类预测 = 命中
            gi = g.reshape(-1).nonzero().flatten()
            pi = p.reshape(-1).nonzero().flatten()
            hit_g = torch.zeros(len(gi), dtype=torch.bool)
            hit_p = torch.zeros(len(pi), dtype=torch.bool)
            for ii, x in enumerate(gi):
                m = (pi >= x - tol) & (pi <= x + tol)
                if m.any():
                    hit_g[ii] = True
                    hit_p |= m
            tp2 = hit_g.sum().item()
            f1t.append(2 * tp2 / max(2 * tp2 + (~hit_p).sum().item() + (len(gi) - tp2), 1))
    seg = out["seg_dir"].argmax(-1).cpu()
    sg = batch["seg_dir"]
    bas_all, bas_conf = [], []
    for s in range(N_SCALES):
        for c in (0, 1, 2):
            m = sg[..., s] == c
            if m.sum() > 0:
                bas_all.append((seg[..., s][m] == c).float().mean().item())
                mc = m & conf[..., s]
                if mc.sum() > 0:
                    bas_conf.append((seg[..., s][mc] == c).float().mean().item())
    r2_all, r2_conf = [], []
    for name in ("bars_since", "amp_since"):
        pred = out[name].cpu().reshape(-1, N_SCALES)
        true = batch[name].reshape(-1, N_SCALES)
        cm = conf.reshape(-1, N_SCALES)
        for s in range(N_SCALES):
            for bucket, mm in ((r2_all, None), (r2_conf, cm[:, s])):
                t, p = (true[:, s], pred[:, s]) if mm is None else (true[mm, s], pred[mm, s])
                if len(t) < 10:
                    continue
                ss_res = ((t - p) ** 2).sum().item()
                ss_tot = ((t - t.mean()) ** 2).sum().item()
                bucket.append(1 - ss_res / max(ss_tot, 1e-9))
    return {"U1_pivF1": sum(f1s) / len(f1s),
            "U2_segBA": sum(bas_all) / len(bas_all),
            "U3_coordR2": sum(r2_all) / len(r2_all),
            "A1_pivF1_tol": sum(f1t) / len(f1t),
            "A2_segBA_conf": sum(bas_conf) / max(len(bas_conf), 1),
            "A2_coordR2_conf": sum(r2_conf) / max(len(r2_conf), 1)}
