"""K-Line Transformer 训练器

极简训练循环，对标 minimind 的训练风格，纯 PyTorch 实现。
新增 Early Stopping。
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from obson.model.transformer import KLineConfig, KLineTransformer


class KLineTrainer:
    """K 线预测模型训练器（含 Early Stopping）"""

    def __init__(
        self,
        model: KLineTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        lr: float = 1e-3,
        weight_decay: float = 0.01,
        max_epochs: int = 100,
        patience: int = 15,           # Early Stopping 耐心值
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_dir: str | Path = "./checkpoints",
        log_interval: int = 10,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_interval = log_interval
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        # OneCycleLR: 先warmup到max_lr，再衰减到接近0，打破学习平台期
        total_steps = max_epochs * len(train_loader)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e4,
        )

        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_state = None       # 保存最佳模型权重（用于早停后恢复）

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = len(self.train_loader)

        for batch_idx, batch in enumerate(self.train_loader):
            x = batch["seq"].to(self.device)
            y = batch["target"].to(self.device)
            temporal = batch.get("temporal_feat")
            hourly = batch.get("hourly_ctx")
            time_pos = batch.get("time_pos")
            freq_feat = batch.get("freq_feat")
            if temporal is not None:
                temporal = temporal.to(self.device)
            if hourly is not None:
                hourly = hourly.to(self.device)
            if time_pos is not None:
                time_pos = time_pos.to(self.device)
            if freq_feat is not None:
                freq_feat = freq_feat.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(x, targets=y, temporal_feat=temporal, hourly_ctx=hourly, time_pos=time_pos, freq_feat=freq_feat)
            loss = out["loss"]
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()  # OneCycleLR 按 step 更新
            total_loss += loss.item()
            self.global_step += 1

            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"  batch [{batch_idx:4d}/{n_batches}] "
                    f"loss={loss.item():.6f} lr={current_lr:.2e}"
                )

        return total_loss / n_batches

    @torch.no_grad()
    def validate(self) -> float:
        if self.val_loader is None:
            return float("inf")

        self.model.eval()
        total_loss = 0.0
        n_batches = len(self.val_loader)

        for batch in self.val_loader:
            x = batch["seq"].to(self.device)
            y = batch["target"].to(self.device)
            temporal = batch.get("temporal_feat")
            hourly = batch.get("hourly_ctx")
            time_pos = batch.get("time_pos")
            freq_feat = batch.get("freq_feat")
            if temporal is not None:
                temporal = temporal.to(self.device)
            if hourly is not None:
                hourly = hourly.to(self.device)
            if time_pos is not None:
                time_pos = time_pos.to(self.device)
            if freq_feat is not None:
                freq_feat = freq_feat.to(self.device)

            out = self.model(x, targets=y, temporal_feat=temporal, hourly_ctx=hourly, time_pos=time_pos, freq_feat=freq_feat)
            total_loss += out["loss"].item()

        return total_loss / n_batches

    def fit(self) -> None:
        print(f"开始训练 | device={self.device} | params={self.model.count_parameters():,}")
        print(f" epochs={self.max_epochs} | patience={self.patience} | train_batches={len(self.train_loader)}")
        if self.val_loader:
            print(f" val_batches={len(self.val_loader)}")
        print("-" * 60)

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()

            epoch_time = time.time() - t0
            print(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"time={epoch_time:.1f}s"
            )

            # 保存最佳模型 + Early Stopping 逻辑
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint("best.pt")
                # 保存最佳权重用于早停后恢复
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                print(f"  → 最佳模型已保存 (val_loss={val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"  patience={self.patience_counter}/{self.patience}")

            # 定期保存
            if epoch % 10 == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")

            # Early Stopping 触发
            if self.patience_counter >= self.patience:
                print(f"\n⏹ Early Stopping 触发！val_loss 连续 {self.patience} 轮未改善。")
                print(f"   最佳 val_loss={self.best_val_loss:.6f} (Epoch {epoch - self.patience})")
                if self.best_state is not None:
                    self.model.load_state_dict(self.best_state)
                    print("   已自动恢复最佳模型权重。")
                break

        print("-" * 60)
        print("训练完成")

    def save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.model.config,
            },
            path,
        )

    def load_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        print(f"已加载检查点: {path}")
