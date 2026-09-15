"""混频训练器 — 同时训练多种频率

核心思路：
- 不同频率 = 不同"语种"
- 每个 step 轮流喂一种频率的 batch
- 模型统一输出 close_pct，自己适应不同尺度
"""

from __future__ import annotations

import time
from itertools import cycle
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from obson.model.transformer import KLineConfig, KLineTransformer


class MixedFrequencyTrainer:
    """混频 K 线预测模型训练器

    同时管理多个频率的 DataLoader，每个 step 轮流训练。
    """

    def __init__(
        self,
        model: KLineTransformer,
        loaders: dict[str, DataLoader],
        val_loaders: dict[str, DataLoader] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        patience: int = 15,
        device: str = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        ),
        save_dir: str | Path = "./checkpoints",
        log_interval: int = 10,
        lr_sched: str = "onecycle",
    ):
        self.model = model.to(device)
        self.loaders = loaders
        self.val_loaders = val_loaders
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 优化器
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # OneCycleLR（默认，冠军配方）；lr_sched="constant" 时禁用调度器（P1 诊断用）
        # 以最大 loader 的步数为一个 epoch（小 loader 循环复用），
        # 保证长历史频率每个 epoch 被完整训练一遍
        self.steps_per_epoch = max(len(ld) for ld in loaders.values())
        # 每个 step 内对每个频率各更新一次，实际步数 = steps_per_epoch × 频率数
        total_steps = max_epochs * self.steps_per_epoch * len(loaders)
        self.lr_sched = lr_sched
        if lr_sched == "onecycle":
            self.scheduler = OneCycleLR(
                self.optimizer, max_lr=lr, total_steps=total_steps,
                pct_start=0.1, div_factor=25, final_div_factor=1e4,
            )
        else:  # constant：固定 lr，不调度
            self.scheduler = None

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state = None

    def _path_loss(self, pl: torch.Tensor, ps: torch.Tensor) -> torch.Tensor:
        """E3 路径状态 masked CE。v1.1：config.path_state_weights 存在时
        逐节点类别加权（治多数类"未触轨"淹没塌缩），否则退化为普通 masked CE。"""
        import torch.nn.functional as F
        w_flat = getattr(self.model.config, "path_state_weights", None)
        if not w_flat:
            return F.cross_entropy(pl.reshape(-1, 3), ps.reshape(-1), ignore_index=-1)
        w = torch.tensor(w_flat, dtype=torch.float32, device=pl.device).view(4, 3)
        losses = []
        for k in range(4):
            if (ps[:, k] == -1).all():
                continue  # 该节点整批 mask，跳过防 NaN
            losses.append(F.cross_entropy(pl[:, k], ps[:, k], weight=w[k], ignore_index=-1))
        return torch.stack(losses).mean() if losses else pl.sum() * 0.0

    def _exc_loss(self, lg_dn: torch.Tensor, lg_up: torch.Tensor, el: torch.Tensor) -> torch.Tensor:
        """E4 excursion 分桶 CE（无 mask，excursion 是事实量）。
        config.exc_weights 存在时逐侧类别加权，否则普通 CE。"""
        import torch.nn.functional as F
        w_flat = getattr(self.model.config, "exc_weights", None)
        if w_flat:
            w = torch.tensor(w_flat, dtype=torch.float32, device=el.device).view(2, 6)
            return (F.cross_entropy(lg_dn, el[:, 0], weight=w[0])
                    + F.cross_entropy(lg_up, el[:, 1], weight=w[1])) / 2
        return (F.cross_entropy(lg_dn, el[:, 0]) + F.cross_entropy(lg_up, el[:, 1])) / 2

    def train_epoch(self) -> dict[str, float]:
        """训练一个 epoch，轮流从各种频率取 batch"""
        self.model.train()

        iters = {freq: cycle(loader) for freq, loader in self.loaders.items()}
        freqs = list(self.loaders.keys())
        steps_per_epoch = self.steps_per_epoch

        losses = {freq: 0.0 for freq in freqs}
        counts = {freq: 0 for freq in freqs}
        # 滑动窗口损失，用于实时日志反映当前真实的收敛速度，而非被远古 batch 稀释
        recent_losses: dict[str, list[float]] = {freq: [] for freq in freqs}

        for step in range(steps_per_epoch):
            for freq in freqs:
                batch = next(iters[freq])
                x = batch["seq"].to(self.device)
                y = batch["target"].to(self.device)
                temporal = batch.get("temporal_feat")
                time_pos = batch.get("time_pos")
                freq_feat = batch.get("freq_feat")
                symbol_id = batch.get("symbol_id")
                daily_ctx = batch.get("daily_ctx")
                if daily_ctx is not None:
                    daily_ctx = daily_ctx.to(self.device)
                foreign_ctx = batch.get("foreign_ctx")
                if foreign_ctx is not None:
                    foreign_ctx = foreign_ctx.to(self.device)
                if temporal is not None:
                    temporal = temporal.to(self.device)
                if time_pos is not None:
                    time_pos = time_pos.to(self.device)
                if freq_feat is not None:
                    freq_feat = freq_feat.to(self.device)
                if symbol_id is not None:
                    symbol_id = symbol_id.to(self.device)

                self.optimizer.zero_grad()
                daily_ctx = batch.get("daily_ctx")
                if daily_ctx is not None:
                    daily_ctx = daily_ctx.to(self.device)
                foreign_ctx = batch.get("foreign_ctx")
                if foreign_ctx is not None:
                    foreign_ctx = foreign_ctx.to(self.device)
                fine_ctx = batch.get("fine_ctx")
                if fine_ctx is not None:
                    fine_ctx = fine_ctx.to(self.device)
                if getattr(self.model.config, "task", "regress") == "classify":
                    labels = batch["label"].to(self.device)
                    soft = batch.get("soft_label")
                    if soft is not None:
                        soft = soft.to(self.device)
                    cross_ctx = batch.get("cross_ctx")
                    cross_mask = batch.get("cross_mask")
                    if cross_ctx is not None:
                        cross_ctx = cross_ctx.to(self.device)
                        cross_mask = cross_mask.to(self.device)
                    utility_targets = batch.get("utility_target")
                    if utility_targets is not None:
                        utility_targets = utility_targets.to(self.device)
                    hazard_targets = batch.get("hazard_states")
                    if hazard_targets is not None:
                        hazard_targets = hazard_targets.to(self.device)
                    # targets 一并传入：dense_loss_weight>0 时分类模式附带逐bar密集监督（表征学习）
                    out = self.model(x, labels=labels, soft_labels=soft, targets=y, temporal_feat=temporal, time_pos=time_pos, freq_feat=freq_feat, symbol_id=symbol_id, daily_ctx=daily_ctx, foreign_ctx=foreign_ctx, cross_ctx=cross_ctx, cross_mask=cross_mask, fine_ctx=fine_ctx, utility_targets=utility_targets, hazard_targets=hazard_targets)
                else:
                    out = self.model(x, targets=y, temporal_feat=temporal, time_pos=time_pos, freq_feat=freq_feat, symbol_id=symbol_id, daily_ctx=daily_ctx, foreign_ctx=foreign_ctx, fine_ctx=fine_ctx)
                loss = out["loss"]
                # E3 路径状态辅助损失（masked CE，-1=同根双触歧义样本整行忽略）
                # 教师模型要求：分项记录 L_main / L_path / λ·L_path，不能只看加权和
                l_main_val = loss.item()
                l_path_val = float("nan")
                if "path_logits" in out and batch.get("path_states") is not None:
                    ps = batch["path_states"].to(self.device)  # [B, 4] long
                    pl = out["path_logits"]                    # [B, 4, 3]
                    l_path = self._path_loss(pl, ps)
                    l_path_val = l_path.item()
                    loss = loss + float(getattr(self.model.config, "path_aux_weight", 0.1)) * l_path
                # E4 excursion 分桶辅助损失
                if "exc_logits_dn" in out and batch.get("exc_labels") is not None:
                    el = batch["exc_labels"].to(self.device)  # [B, 2] long
                    l_exc = self._exc_loss(out["exc_logits_dn"], out["exc_logits_up"], el)
                    self._last_l_exc = l_exc.item()
                    loss = loss + float(getattr(self.model.config, "exc_aux_weight", 0.1)) * l_exc
                self._last_l_main, self._last_l_path = l_main_val, l_path_val
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

                l_val = loss.item()
                losses[freq] += l_val
                counts[freq] += 1
                recent_losses[freq].append(l_val)
                if len(recent_losses[freq]) > 20:
                    recent_losses[freq].pop(0)
                self.global_step += 1

                if step % self.log_interval == 0 and freq == freqs[-1]:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    loss_str = " ".join(
                        f"| {f}={sum(recent_losses[f])/len(recent_losses[f]):.4f}" for f in freqs
                    )
                    path_str = ""
                    lp = getattr(self, "_last_l_path", float("nan"))
                    if lp == lp:  # not NaN → path_aux 开启，分项打印（教师模型要求）
                        w = float(getattr(self.model.config, "path_aux_weight", 0.1))
                        path_str = f" | L_main={self._last_l_main:.4f} L_path={lp:.4f} λL_path={w*lp:.4f}"
                    print(
                        f"  step [{step:4d}/{steps_per_epoch}] "
                        f"{loss_str} "
                        f"| lr={current_lr:.2e}{path_str}"
                    )

        return {freq: losses[freq] / max(counts[freq], 1) for freq in freqs}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        if self.val_loaders is None:
            return {}

        self.model.eval()
        result = {}
        ics = {}
        cls_metrics = {}
        task = getattr(self.model.config, "task", "regress")
        for freq, loader in self.val_loaders.items():
            total_loss = 0.0
            n_batches = len(loader)
            preds, trues, fwds = [], [], []
            logits_l, thetas_l = [], []
            path_pred_l, path_true_l = [], []  # E3：验证集路径状态逐节点指标
            exc_pred_l, exc_true_l = [], []    # E4：验证集 excursion 分桶指标
            utility_l, utility_target_l = [], []
            hazard_prob_l, hazard_target_l = [], []
            hazard_agg_l = []
            for batch in loader:
                x = batch["seq"].to(self.device)
                y = batch["target"].to(self.device)
                labels = batch.get("label")
                temporal = batch.get("temporal_feat")
                time_pos = batch.get("time_pos")
                freq_feat = batch.get("freq_feat")
                symbol_id = batch.get("symbol_id")
                daily_ctx = batch.get("daily_ctx")
                if daily_ctx is not None:
                    daily_ctx = daily_ctx.to(self.device)
                foreign_ctx = batch.get("foreign_ctx")
                if foreign_ctx is not None:
                    foreign_ctx = foreign_ctx.to(self.device)
                fine_ctx = batch.get("fine_ctx")
                if fine_ctx is not None:
                    fine_ctx = fine_ctx.to(self.device)
                if temporal is not None:
                    temporal = temporal.to(self.device)
                if time_pos is not None:
                    time_pos = time_pos.to(self.device)
                if freq_feat is not None:
                    freq_feat = freq_feat.to(self.device)
                if symbol_id is not None:
                    symbol_id = symbol_id.to(self.device)
                if task == "classify":
                    if labels is None:
                        continue
                    labels = labels.to(self.device)
                    cross_ctx = batch.get("cross_ctx")
                    cross_mask = batch.get("cross_mask")
                    if cross_ctx is not None:
                        cross_ctx = cross_ctx.to(self.device)
                        cross_mask = cross_mask.to(self.device)
                    utility_targets = batch.get("utility_target")
                    if utility_targets is not None:
                        utility_targets = utility_targets.to(self.device)
                    hazard_targets = batch.get("hazard_states")
                    if hazard_targets is not None:
                        hazard_targets = hazard_targets.to(self.device)
                    out = self.model(x, labels=labels, temporal_feat=temporal, time_pos=time_pos, freq_feat=freq_feat, symbol_id=symbol_id, daily_ctx=daily_ctx, foreign_ctx=foreign_ctx, cross_ctx=cross_ctx, cross_mask=cross_mask, fine_ctx=fine_ctx, utility_targets=utility_targets, hazard_targets=hazard_targets)
                    v_loss = out["loss"]
                    if "hazard_logits" in out and hazard_targets is not None:
                        hazard_prob_l.append(out["hazard_logits"].softmax(-1).float().cpu())
                        hazard_target_l.append(hazard_targets.float().cpu())
                        # Public trading probabilities are the aggregated
                        # [down, none, up] distribution, not per-bin hazards.
                        hazard_agg_l.append(out["logits"].exp().float().cpu())
                    if "loss_utility" in out:
                        utility_l.append(out["utility_scores"].float().cpu())
                        utility_target_l.append(utility_targets.float().cpu())
                    # E3：验证集同样计入路径辅助损失，保持 train/val 损失口径一致
                    if "path_logits" in out and batch.get("path_states") is not None:
                        ps = batch["path_states"].to(self.device)
                        v_loss = v_loss + float(getattr(self.model.config, "path_aux_weight", 0.1)) * self._path_loss(
                            out["path_logits"], ps)
                        path_pred_l.append(out["path_logits"].argmax(dim=-1).float().cpu())
                        path_true_l.append(ps.float().cpu())
                    # E4：验证集同样计入 excursion 辅助损失，保持口径一致
                    if "exc_logits_dn" in out and batch.get("exc_labels") is not None:
                        el = batch["exc_labels"].to(self.device)
                        v_loss = v_loss + float(getattr(self.model.config, "exc_aux_weight", 0.1)) * self._exc_loss(
                            out["exc_logits_dn"], out["exc_logits_up"], el)
                        exc_pred_l.append(torch.stack([out["exc_logits_dn"].argmax(-1),
                                                       out["exc_logits_up"].argmax(-1)], dim=1).float().cpu())
                        exc_true_l.append(el.float().cpu())
                    total_loss += v_loss.item()
                    preds.append(out["pred_class"].float().cpu())
                    logits_l.append(out["logits"].float().cpu())
                    trues.append(labels.float().cpu())
                    fwd = batch.get("fwd_ret")
                    if fwd is not None:
                        fwds.append(fwd.float().cpu())
                    th_b = batch.get("theta")
                    if th_b is not None:
                        thetas_l.append(th_b.float().cpu())
                else:
                    out = self.model(x, targets=y, temporal_feat=temporal, time_pos=time_pos, freq_feat=freq_feat, symbol_id=symbol_id, daily_ctx=daily_ctx, foreign_ctx=foreign_ctx, fine_ctx=fine_ctx)
                    total_loss += out["loss"].item()
                    base = x[:, -1, 3]
                    true_pct = (y[:, -1, 3] - base) / (base.abs() + 1e-8) * 100
                    preds.append(out["close_pct"].float().cpu())
                    trues.append(true_pct.float().cpu())
            result[freq] = total_loss / max(n_batches, 1)
            if not preds:
                continue
            p = torch.cat(preds).numpy()
            t = torch.cat(trues).numpy()
            import numpy as np
            if task == "classify":
                # balanced accuracy + 信号类精确率/覆盖率
                recalls, precs = [], {}
                for c in (0, 1, 2):
                    tp = float(((p == c) & (t == c)).sum())
                    fn = float(((p != c) & (t == c)).sum())
                    fp = float(((p == c) & (t != c)).sum())
                    recalls.append(tp / max(tp + fn, 1.0))
                    precs[c] = tp / max(tp + fp, 1.0)
                cls_metrics[freq] = {
                    "bal_acc": float(np.mean(recalls)),
                    "recall_neg": recalls[0], "recall_none": recalls[1], "recall_pos": recalls[2],
                    "prec_pos": precs[2], "prec_neg": precs[0],
                    "cover": float((p != 1).mean()),
                }
                # 实战 edge：喊开命中率超出"乱喊基准"的幅度，按信号数做 shrinkage，
                # 防止"只喊 1 次碰巧喊对"拿到虚高 edge。选模与早停的唯一裁判。
                base_pos = float((t == 2).mean())
                base_neg = float((t == 0).mean())
                n_pos = float((p == 2).sum())
                n_neg = float((p == 0).sum())
                edge = (precs[2] - base_pos) * min(1.0, n_pos / 50.0) \
                     + (precs[0] - base_neg) * min(1.0, n_neg / 50.0)
                cls_metrics[freq]["edge"] = float(edge)
                if hazard_prob_l and hazard_target_l:
                    hp = torch.cat(hazard_prob_l).numpy()
                    ht = torch.cat(hazard_target_l).numpy().astype(int)
                    valid_h = ht >= 0
                    cls_metrics[freq]["hazard_survival_mass"] = float(hp[..., 0].mean())
                    cls_metrics[freq]["hazard_up_mass"] = float(hp[..., 1].mean())
                    cls_metrics[freq]["hazard_down_mass"] = float(hp[..., 2].mean())
                    cls_metrics[freq]["hazard_event_rate"] = float(
                        ((ht == 1) | (ht == 2)).sum() / max(valid_h.sum(), 1)
                    )
                    cls_metrics[freq]["hazard_event_mass"] = float(hp[..., 1:].sum(-1).mean())
                if hazard_agg_l:
                    ap = torch.cat(hazard_agg_l).numpy()
                    cls_metrics[freq]["hazard_p_down"] = float(ap[:, 0].mean())
                    cls_metrics[freq]["hazard_p_none"] = float(ap[:, 1].mean())
                    cls_metrics[freq]["hazard_p_up"] = float(ap[:, 2].mean())
                    cls_metrics[freq]["hazard_final_event_mass"] = float(
                        (ap[:, 0] + ap[:, 2]).mean()
                    )
                    cls_metrics[freq]["hazard_argmax_cover"] = float(
                        (ap.argmax(axis=1) != 1).mean()
                    )
                # E3 路径辅助头验证指标（教师模型 §8）：逐节点 BA + 单调性违规率 + 塌缩检测
                if path_pred_l:
                    pp = torch.cat(path_pred_l).numpy().astype(int)   # [N,4]
                    pt = torch.cat(path_true_l).numpy().astype(int)
                    valid_rows = (pt != -1).any(axis=1)
                    pp, pt = pp[valid_rows], pt[valid_rows]
                    node_ba, node_acc, node_f1, node_true_dist, node_rec = [], [], [], [], []
                    for k in range(4):
                        mk = pt[:, k] != -1
                        if mk.sum() == 0:
                            node_ba.append(float("nan")); node_acc.append(float("nan"))
                            node_f1.append(float("nan")); node_true_dist.append([float("nan")]*3)
                            node_rec.append([float("nan")]*3); continue
                        p_k, t_k = pp[mk, k], pt[mk, k]
                        rc, f1s = [], []
                        for c in (0, 1, 2):
                            tp = float(((p_k == c) & (t_k == c)).sum())
                            fn = float(((p_k != c) & (t_k == c)).sum())
                            fp = float(((p_k == c) & (t_k != c)).sum())
                            r = tp / max(tp + fn, 1.0)
                            pr = tp / max(tp + fp, 1.0)
                            rc.append(r)
                            f1s.append(2 * pr * r / max(pr + r, 1e-12))
                        node_rec.append(rc)
                        node_ba.append(float(np.mean(rc)))
                        node_f1.append(float(np.mean(f1s)))  # macro F1（教师模型要求）
                        node_acc.append(float((p_k == t_k).mean()))
                        # 真实标签分布：判断 BA 增益必须对照多数类规模
                        node_true_dist.append([float((t_k == c).mean()) for c in (0, 1, 2)])
                    # 单调性违规：预测序列出现 触轨(1/2)后又回到未触轨(0)，或两侧互跳
                    viol = 0
                    for row in pp:
                        seen = 0
                        for st in row:
                            if st in (1, 2):
                                if seen != 0 and st != seen:
                                    viol += 1; break
                                seen = st
                            elif st == 0 and seen != 0:
                                viol += 1; break
                    cls_metrics[freq]["path_node_ba"] = node_ba
                    cls_metrics[freq]["path_node_acc"] = node_acc
                    cls_metrics[freq]["path_node_f1"] = node_f1
                    cls_metrics[freq]["path_node_true_dist"] = node_true_dist
                    cls_metrics[freq]["path_node_recall"] = node_rec
                    cls_metrics[freq]["path_mono_viol"] = viol / max(len(pp), 1)
                    # 塌缩检测：预测状态分布（全押"未触轨"或复制主标签都属于塌缩）
                    cls_metrics[freq]["path_pred_dist"] = [float((pp == c).mean()) for c in (0, 1, 2)]
                # E4 excursion 分桶验证指标：逐侧 BA / acc / 预测分布（塌缩检测）
                if exc_pred_l:
                    ep = torch.cat(exc_pred_l).numpy().astype(int)  # [N,2]
                    et = torch.cat(exc_true_l).numpy().astype(int)
                    exc_ba, exc_acc, exc_dist = [], [], []
                    for side in range(2):
                        p_s, t_s = ep[:, side], et[:, side]
                        rc = [float(((p_s == c) & (t_s == c)).sum()) / max(float((t_s == c).sum()), 1.0) for c in range(6)]
                        exc_ba.append(float(np.mean(rc)))
                        exc_acc.append(float((p_s == t_s).mean()))
                        exc_dist.append([float((p_s == c).mean()) for c in range(6)])
                    cls_metrics[freq]["exc_ba"] = exc_ba
                    cls_metrics[freq]["exc_acc"] = exc_acc
                    cls_metrics[freq]["exc_pred_dist"] = exc_dist
                if fwds:
                    # 实战含义：val 上喊多/喊空后的平均实际收益(%)，选模的经济价值参照
                    fwd_arr = torch.cat(fwds).numpy()
                    cls_metrics[freq]["ret_pos"] = float(fwd_arr[p == 2].mean()) if (p == 2).any() else float("nan")
                    cls_metrics[freq]["ret_neg"] = float(fwd_arr[p == 0].mean()) if (p == 0).any() else float("nan")
                # policy_edge（老师审查 #5 修复）：与生产口径同构的辅助指标。
                # 固定规则形式防过拟合：每方向 top-5/100 阈值（每 epoch 仅重算分位数）
                # + 手册v1方向过滤 + 近似结算（命中=+0.8θ / 反向=−0.5θ / 无=fwd_ret）− 双边成本。
                # 只打印、不参与选模/早停（选模裁判仍是 argmax mean_edge）。
                if fwds and thetas_l:
                    from obson.playbook import PLAYBOOK_V1, COST_RT
                    lg = torch.cat(logits_l).numpy()
                    lg = lg - lg.max(axis=1, keepdims=True)
                    prob = np.exp(lg) / np.exp(lg).sum(axis=1, keepdims=True)
                    th_arr = torch.cat(thetas_l).numpy()
                    side = PLAYBOOK_V1.get(freq)
                    code = freq.rsplit("_", 1)[0]
                    cost = COST_RT.get(code, 0.06)
                    pol_pnls, pol_n = [], 0
                    for cls in (2, 0):
                        if side == "long" and cls == 0:
                            continue
                        if side == "short" and cls == 2:
                            continue
                        n = len(prob)
                        k = max(int(n * 5 / 100), 5)
                        top = np.sort(prob[:, cls])[::-1]
                        thr_c = float(top[min(k - 1, n - 1)])
                        sig = prob[:, cls] >= thr_c
                        if not sig.any():
                            continue
                        hit = t == cls
                        opp = t == (0 if cls == 2 else 2)
                        pnl = np.where(hit, 0.8 * th_arr,
                                       np.where(opp, -0.5 * th_arr,
                                                fwd_arr if cls == 2 else -fwd_arr))
                        pol_pnls.append(pnl[sig] - cost)
                        pol_n += int(sig.sum())
                    if pol_pnls:
                        pp = np.concatenate(pol_pnls)
                        cls_metrics[freq]["policy_edge"] = float(pp.mean())
                        cls_metrics[freq]["policy_n"] = int(pol_n)
                if utility_l and utility_target_l:
                    us = torch.cat(utility_l).numpy()
                    ut = torch.cat(utility_target_l).numpy()
                    chosen = us.argmax(axis=1)
                    chosen_score = us[np.arange(len(us)), chosen]
                    # 只评价高置信效用排序的 top 5%，避免效用头靠全量噪声取平均。
                    k = max(int(len(us) * 0.05), 5)
                    keep = np.argsort(chosen_score)[-min(k, len(us)):]
                    cls_metrics[freq]["utility_edge"] = float(
                        ut[keep, chosen[keep]].mean()
                    ) if len(keep) else float("nan")
                    cls_metrics[freq]["utility_cover"] = float(len(keep) / max(len(us), 1))
            else:
                ics[freq] = float(np.corrcoef(p, t)[0, 1]) if len(p) > 2 else float("nan")
        self._last_val_ics = ics
        self._last_val_cls = cls_metrics
        return result

    def _compute_cls_baselines(self) -> dict[str, float]:
        """classify 任务的先验基线：只按类别先验分布出牌的 CE"""
        baselines = {}
        for freq, loader in (self.val_loaders or {}).items():
            counts = torch.zeros(3)
            for batch in loader:
                labels = batch.get("label")
                if labels is not None:
                    counts += torch.bincount(labels.view(-1), minlength=3).float()
            p = (counts / counts.sum().clamp(min=1)).clamp(min=1e-8)
            baselines[freq] = float(-(p * p.log()).sum())
        return baselines

    def _compute_val_baselines(self) -> dict[str, float]:
        """各组合的"零预测"基线 val loss —— 用于归一化模型选择分数

        不同品种/频率的目标波动量级不同（铁矿 1h std≈0.64%，热卷≈0.37%），
        直接平均原始 Huber loss 会让模型选择被高波动品种绑架。
        """
        baselines = {}
        beta = getattr(self.model.config, "huber_delta", 1.0)
        with torch.no_grad():
            for freq, loader in (self.val_loaders or {}).items():
                losses = []
                for batch in loader:
                    x, y = batch["seq"], batch["target"]
                    base = x[:, -1, 3]
                    true_pct = (y[:, -1, 3] - base) / (base.abs() + 1e-8) * 100
                    z = torch.zeros_like(true_pct)
                    losses.append(torch.nn.functional.smooth_l1_loss(z, true_pct, beta=beta).item())
                baselines[freq] = sum(losses) / max(len(losses), 1)
        return baselines

    def fit(self) -> None:
        freqs = list(self.loaders.keys())
        task = getattr(self.model.config, "task", "regress")
        self.val_baselines = self._compute_val_baselines() if task == "regress" else self._compute_cls_baselines()
        print(f"开始混频训练 | device={self.device} | params={self.model.count_parameters():,}")
        print(f"  frequencies={freqs}")
        print(f"  epochs={self.max_epochs} | patience={self.patience} | lr_sched={self.lr_sched}")
        bl_str = " ".join(f"{f}={self.val_baselines.get(f, 0):.4f}" for f in freqs)
        print(f"  {'零预测基线 val loss' if task == 'regress' else '先验分布基线 CE'}: {bl_str}")
        print("-" * 70)

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_losses = self.train_epoch()
            val_losses = self.validate()

            avg_val_loss = sum(val_losses.values()) / len(val_losses) if val_losses else float("inf")
            # 归一化分数：每个组合除以各自的零预测基线再平均，消除品种/频率间的量级差
            norm_score = sum(
                val_losses[f] / self.val_baselines.get(f, 1.0) for f in val_losses
            ) / len(val_losses) if val_losses else float("inf")

            # 模型选择对最大化的指标：regress=验证集平均 IC；
            # classify=验证集平均实战 edge（喊多/喊空命中率超出各自基准的幅度之和，
            # 按信号数 shrinkage）。选模与早停共用此分数。
            import numpy as np
            if task == "classify":
                score_vals = [m["edge"] for m in self._last_val_cls.values()]
                selection_score = float(np.mean(score_vals)) if score_vals else float("-inf")
                score_name = "mean_edge"
                if getattr(self.model.config, "utility_head", False):
                    utility_vals = [m.get("utility_edge") for m in self._last_val_cls.values()
                                    if m.get("utility_edge") is not None]
                    if utility_vals:
                        utility_score = float(np.nanmean(utility_vals))
                        uw = float(getattr(self.model.config, "utility_selection_weight", 0.10))
                        selection_score += uw * utility_score
                        score_name = "teacher_score"
                        print(f"  teacher utility_edge={utility_score:+.4f} weight={uw:.3f}")
            else:
                ic_vals = [v for v in self._last_val_ics.values() if not np.isnan(v)]
                selection_score = float(np.mean(ic_vals)) if ic_vals else float("-inf")
                score_name = "mean_ic"

            epoch_time = time.time() - t0
            train_str = " ".join(f"{f}={train_losses[f]:.4f}" for f in freqs)
            val_str = " ".join(f"{f}={val_losses.get(f, 0):.4f}" for f in freqs)
            if task == "classify":
                ic_str = " ".join(
                    f"{f}={self._last_val_cls.get(f, {}).get('bal_acc', float('nan')):.3f}"
                    for f in freqs
                )
                extra_str = " ".join(
                    f"{f}:r-={self._last_val_cls.get(f, {}).get('recall_neg', 0):.2f}/"
                    f"r0={self._last_val_cls.get(f, {}).get('recall_none', 0):.2f}/"
                    f"r+={self._last_val_cls.get(f, {}).get('recall_pos', 0):.2f}/"
                    f"p+={self._last_val_cls.get(f, {}).get('prec_pos', 0):.2f}/"
                    f"p-={self._last_val_cls.get(f, {}).get('prec_neg', 0):.2f}/"
                    f"cov={self._last_val_cls.get(f, {}).get('cover', 0):.0%}/"
                    f"r+={self._last_val_cls.get(f, {}).get('ret_pos', 0):+.3f}%/"
                    f"r-={self._last_val_cls.get(f, {}).get('ret_neg', 0):+.3f}%"
                    for f in freqs
                )
                # 生产口径辅助指标（policy_edge）：只打印不参选模
                pol = {f: self._last_val_cls.get(f, {}).get("policy_edge") for f in freqs}
                pol = {f: v for f, v in pol.items() if v is not None}
                if pol:
                    extra_str += "\n  policy_edge(每100根喊5次+手册v1+近似结算, 不参与选模): " + " ".join(
                        f"{f}={v:+.4f}%" for f, v in pol.items())
                # E3 路径辅助头指标（教师模型 §8 + 中期评审）：只打印不参选模
                pm = [m for m in self._last_val_cls.values() if "path_node_ba" in m]
                if pm:
                    nba = np.nanmean([m["path_node_ba"] for m in pm], axis=0)
                    nacc = np.nanmean([m["path_node_acc"] for m in pm], axis=0)
                    nf1 = np.nanmean([m["path_node_f1"] for m in pm], axis=0)
                    td = np.nanmean([m["path_node_true_dist"] for m in pm], axis=0)
                    rcl = np.nanmean([m["path_node_recall"] for m in pm], axis=0)
                    viol = float(np.mean([m["path_mono_viol"] for m in pm]))
                    dist = np.mean([m["path_pred_dist"] for m in pm], axis=0)
                    extra_str += (
                        "\n  path_aux(辅助头, 不参与选模): "
                        + "节点BA=[" + "/".join(f"{v:.3f}" for v in nba) + "]"
                        + " F1=[" + "/".join(f"{v:.2f}" for v in nf1) + "]"
                        + " acc=[" + "/".join(f"{v:.2f}" for v in nacc) + "]"
                        + f" 单调违规={viol:.1%} 预测分布(未/上/下)=[{dist[0]:.2f}/{dist[1]:.2f}/{dist[2]:.2f}]"
                        + "\n    真实分布=[" + "/".join(f"({r[0]:.2f},{r[1]:.2f},{r[2]:.2f})" for r in td) + "]"
                        + " recall上/下=[" + "/".join(f"({r[1]:.2f},{r[2]:.2f})" for r in rcl) + "]"
                    )
                # E4 excursion 指标：只打印不参选模
                em = [m for m in self._last_val_cls.values() if "exc_ba" in m]
                if em:
                    eba = np.mean([m["exc_ba"] for m in em], axis=0)
                    eacc = np.mean([m["exc_acc"] for m in em], axis=0)
                    edn = np.mean([m["exc_pred_dist"][0] for m in em], axis=0)
                    eup = np.mean([m["exc_pred_dist"][1] for m in em], axis=0)
                    extra_str += (
                        f"\n  exc_aux(辅助头, 不参与选模): BA(下/上)=[{eba[0]:.3f}/{eba[1]:.3f}]"
                        f" acc=[{eacc[0]:.2f}/{eacc[1]:.2f}]"
                        f" 分布dn=[" + "/".join(f"{v:.2f}" for v in edn) + "]"
                        f" up=[" + "/".join(f"{v:.2f}" for v in eup) + "]"
                    )
                hm = [m for m in self._last_val_cls.values() if "hazard_event_mass" in m]
                if hm:
                    extra_str += (
                        "\n  hazard(不参与选模): "
                        f"survival={np.mean([m['hazard_survival_mass'] for m in hm]):.3f} "
                        f"event_mass={np.mean([m['hazard_event_mass'] for m in hm]):.3f} "
                        f"true_event={np.mean([m['hazard_event_rate'] for m in hm]):.3f} "
                        f"final(P空/无/多)="
                        f"{np.mean([m.get('hazard_p_down', float('nan')) for m in hm]):.3f}/"
                        f"{np.mean([m.get('hazard_p_none', float('nan')) for m in hm]):.3f}/"
                        f"{np.mean([m.get('hazard_p_up', float('nan')) for m in hm]):.3f} "
                        f"final_event={np.mean([m.get('hazard_final_event_mass', float('nan')) for m in hm]):.3f} "
                        f"argmax_cover={np.mean([m.get('hazard_argmax_cover', float('nan')) for m in hm]):.1%}"
                    )
            else:
                ic_str = " ".join(f"{f}={self._last_val_ics.get(f, float('nan')):+.3f}" for f in freqs)
                extra_str = ""
            ba_mean = float(np.mean([m["bal_acc"] for m in self._last_val_cls.values()])) \
                if task == "classify" and self._last_val_cls else float("nan")
            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | time={epoch_time:.1f}s\n"
                f"  train: {train_str}\n"
                f"  val:   {val_str} (avg={avg_val_loss:.4f} | norm={norm_score:.4f})\n"
                f"  val {'BA' if task == 'classify' else 'IC'}: {ic_str}"
                + (f" (mean_BA={ba_mean:+.4f} | {score_name}={selection_score:+.4f})\n"
                   if task == "classify" else f" (mean={selection_score:+.4f})\n")
                + (f"  {extra_str}\n" if extra_str else "")
            )

            if selection_score > getattr(self, "_best_score", float("-inf")):
                self._best_score = selection_score
                self.best_val_loss = avg_val_loss
                self.patience_counter = 0
                self.save_checkpoint("best.pt")
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"  → 最佳模型已保存 ({score_name}={selection_score:+.4f}, avg_val_loss={avg_val_loss:.4f})")
            else:
                self.patience_counter += 1
                print(f"  patience={self.patience_counter}/{self.patience}")

            if epoch % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

            if self.patience_counter >= self.patience:
                print(f"\n⏹ Early Stopping 触发！最佳 {score_name}={getattr(self, '_best_score', float('nan')):+.4f}")
                if self.best_state is not None:
                    self.model.load_state_dict(self.best_state)
                    print("  已自动恢复最佳模型权重。")
                break

        print("-" * 70)
        print("训练完成")

    def save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.model.config,
            # 完整数据/标签配置清单（老师二轮 #5）：ckpt 自解释训练口径
            "data_manifest": getattr(self, "data_manifest", None),
        }, path)

    def load_checkpoint(self, filename: str) -> None:
        path = Path(filename)
        if not path.is_absolute() and str(path.parent) == ".":
            # 纯文件名，拼接 save_dir
            path = self.save_dir / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.scheduler is not None and ckpt.get("scheduler") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(f"已加载检查点: {path}")
