"""K-Line Transformer 模型 v12 — 技术指标改因果扩展窗口，无 volume（kline_dim=4）

改动：
1. raw_proj 仅接收 OHLC 相对 close 的百分比（×100），彻底去掉 volume
2. 预测目标改为 close 百分比变化（close_pct）
3. Feature Encoder 保持 4 分支：raw + kline_feat + temporal + technical
4. 修复 KLineFeatureEncoder 重复定义问题
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class KLineConfig:
    kline_dim: int = 4
    hidden_size: int = 32
    num_hidden_layers: int = 1
    num_attention_heads: int = 2
    num_key_value_heads: int = 1
    head_dim: int = 16
    intermediate_size: int = 64
    max_position_embeddings: int = 512
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    loss_type: str = "huber"
    huber_delta: float = 1.0
    dense_loss_weight: float = 0.0  # 全位置监督权重（实验证明为负优化，默认关闭，仅保留代码路径）
    dense_tail: int = 16            # 只监督窗口末尾 N 个位置，前端上下文不足的 token 不监督
    num_symbols: int = 1            # 品种数（多品种训练时的 embedding 表大小）
    use_technical: bool = True      # 实验开关：False=technical分支输出置零（消融实验用）
    task: str = "regress"           # "regress"=回归(预测涨跌幅)  "classify"=三分类(正/无/负信号)
    num_classes: int = 3
    class_weights: list | None = None  # 类别不平衡权重 [负, 无, 正]
    use_alibi: bool = False         # False=标准causal attention + learned pos emb；True=ALiBi
    daily_bars: int = 0             # >0：日线上下文金字塔（N根完整日K作为前缀token）
    foreign_bars: int = 0           # >0：外盘日线上下文（N根外盘收盘特征token，拼在内盘日K之后）
    cross_daily_bars: int = 0       # >0：跨品种日线（iTransformer式 variate token：
                                    # 每品种N根已收盘日K编码→池化为1个品种token，
                                    # 品种token间做一层双向attention，再作前缀拼入主干）
    cross_symbol_ids: list | None = None  # cross_ctx 的品种id顺序（None=arange(num_symbols)）
    sector_groups: list | None = None   # 板块分组 token（iTransformer variate 的轻量版）：
                                        # [[黑色系ids],[油脂ids],...]；每品种日K经共享 DailyContextEncoder
                                        # 编码池化后，组内带mask均值 → 每板块1个token，零新增时序参数、
                                        # 无跨品种attention（cross配方的方差来源被切除）
    close_path: bool = False        # True=价格通道全列锚定窗口前3根收盘均值（路径信息）；False=原版相对自身收盘
    use_vol_oi: bool = False        # True=成交量/持仓量分支（log1p 后 Z-Score 的第5/6列：
                                    # volume=当前bar量能水平, close_oi=持仓量；模型侧做量比/持仓变化特征）
    bidirectional: bool = False     # True=窗口内双向 attention（分类信号合法：预测点在窗口末，
                                    # 全部输入都在过去，无未来泄露；几何形状识别更完整。
                                    # 注意：双向下逐bar密集辅助监督(dense)失效，勿同开）
    rope: bool = False              # True=时间感知 RoPE：旋转角度=真实流逝时间（time_pos），
                                    # 隔夜跳空两侧自动拉开距离；分钟段用 time_pos，
                                    # 日K/外盘前缀按 1 bar 单位均匀排布在窗口起点前
    readout: str = "last"           # 分类读出位置：last=最后位置（单向标配）；
                                    # mean=全局平均；cls=ViT 式 [CLS] token（建议配 bidirectional）
    margin_weight: float = 0.0      # >0：方向间隔损失权重——要求赢家方向 logit 拉开输家 margin 以上，
                                    # 骑墙（多≈空）在有方向证据的样本上受罚；只管排序不管绝对概率
    margin: float = 1.0             # 方向间隔的 logit 距离门槛
    fine_bars: int = 0              # >0：细粒度上下文（5/15m 最近 N 根已收盘 bar 编码为 token，
                                    # 拼在分钟主窗口之前；只作输入不作目标，因果闸在 dataset 侧）
    clean_path: bool = False        # True=干净路径标签：摸轨前逆向偏移>clean_frac×θ 的方向样本
                                    # 改判"按兵不动"（仅影响标签语义，模型结构不变；评测脚本需据此复现标签）
    clean_frac: float = 0.5         # 干净路径的逆向容差（θ 的倍数）
    strategy_label: bool = False    # True=策略对齐标签：多=先摸+tp_frac×θ且未先破-stop_frac×θ，
                                    # 空镜像，其余=无（让训练目标=实盘打法的胜负定义；与 clean_path 互斥）
    tp_frac: float = 0.8            # 策略标签止盈轨（θ 的倍数）
    stop_frac: float = 0.5          # 策略标签止损轨（θ 的倍数）
    theta_q: float | None = None    # 训练时的 θ 分位数（标签宽度口径；消费脚本应从这里恢复，
                                    # 禁止靠命令行记忆传递 —— 老师审查 #6 修复）
    soft_label: bool = False        # 训练时是否用软标签（影响评测脚本复现标签语义）
    contract_mode: bool = False     # 合约模式训练（数据来自 data/contracts 段帧；
                                    # 消费脚本必须匹配数据口径，主连数据+合约模型=错配）
    path_aux: bool = False          # E3 路径状态辅助任务：4 节点(25/50/75/100%剩余)
                                    # × 3 态(未触轨/已上轨/已下轨)，pooled 共享表示挂辅助头
    path_aux_weight: float = 0.1    # 辅助损失权重（masked CE，同根双触节点 mask）
    path_state_weights: list | None = None  # E3 v1.1：逐节点类别权重 [4节点×3态] 展平，
                                    # 训练集状态频率求逆（治 90%+ "未触轨"多数类淹没导致的塌缩）
    exc_aux: bool = False           # E4 excursion 分桶辅助任务：m_dn/m_up 各 6 桶
                                    # （桶边 [0.25,0.5,0.8,1.0,1.5]×θ，pooled 挂两个头）
    exc_aux_weight: float = 0.1     # excursion 辅助损失权重
    exc_weights: list | None = None  # E4：逐侧类别权重 [2侧×6桶] 展平，训练集桶频率求逆
    query_decoder: bool = False     # E6'：未来时间 query decoder —— 4 个可学习 query
                                    # （未来25/50/75/100%时点）cross-attend 历史 encoder 输出，
                                    # 替代 pooled+node_emb 的简易路径头；encoder 保持单向

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size is None:
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight * (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps))).type_as(x)


def causal_ma(x: torch.Tensor, w: int) -> torch.Tensor:
    """因果滑动均值：位置 t 用最近 min(w, t+1) 根 bar。

    窗口前端的 bar 历史不足时，用已有的真实 bar 算扩展均值，
    而不是用复制第一根凑数（复制填充会让 MA5/MA20/ATR14 在窗口前段失真）。
    """
    s = x.shape[-1]
    x_pad = F.pad(x.unsqueeze(1), (w - 1, 0))               # 左侧补 0
    sums = F.avg_pool1d(x_pad, w, stride=1).squeeze(1) * w  # 还原成滚动和
    counts = torch.arange(1, s + 1, device=x.device, dtype=x.dtype).clamp(max=w)
    return sums / counts


def _get_alibi_slopes(n_heads: int) -> list[float]:
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start ** i) for i in range(n)]
    if math.log2(n_heads).is_integer():
        return get_slopes_power_of_2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    slopes = get_slopes_power_of_2(closest)
    extra = get_slopes_power_of_2(2 * closest)
    extra = extra[0::2][: n_heads - closest]
    slopes.extend(extra)
    return slopes


class CausalAttention(nn.Module):
    """标准 causal attention，支持可选 ALiBi bias。"""
    def __init__(self, config: KLineConfig):
        super().__init__()
        self.use_alibi = config.use_alibi
        self.bidirectional = getattr(config, "bidirectional", False)
        self.rope = getattr(config, "rope", False)
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads or config.num_attention_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        if self.rope:
            # RoPE 逆频率（标准 base=10000）；角度 = 位置 × inv_freq
            inv = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
            self.register_buffer("rope_inv_freq", inv, persistent=False)
        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if self.use_alibi:
            slopes = torch.tensor(_get_alibi_slopes(self.n_local_heads))
            self.slope_log = nn.Parameter(slopes.log())
        self.flash = hasattr(F, "scaled_dot_product_attention")

    def _slopes(self) -> torch.Tensor:
        return torch.exp(self.slope_log).view(1, -1, 1, 1)

    def _make_alibi_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        q_pos = torch.arange(seq_len, device=device).view(-1, 1).float()
        k_pos = torch.arange(seq_len, device=device).view(1, -1).float()
        distances = (q_pos - k_pos).clamp(min=0).float().unsqueeze(0).unsqueeze(0)
        bias = -self._slopes().to(device) * distances
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
        return bias.masked_fill(causal_mask, float("-inf"))

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                    positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """时间感知旋转：positions [B, S] 为真实流逝时间（bar 间隔中位数=1 单位）。
        q/k: [B, H, S, D]。隔夜/休市的真实时间差直接反映在旋转角差上。"""
        freqs = positions.float().unsqueeze(-1) * self.rope_inv_freq  # [B, S, D/2]
        cos = freqs.cos().unsqueeze(1)  # [B, 1, S, D/2]
        sin = freqs.sin().unsqueeze(1)

        def rot(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return rot(q), rot(k)

    def forward(self, x: torch.Tensor, time_positions: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=1)
            xv = xv.repeat_interleave(self.n_rep, dim=1)
        if self.rope and time_positions is not None:
            xq, xk = self._apply_rope(xq, xk, time_positions)
        if self.use_alibi:
            attn_mask = self._make_alibi_mask(seq_len, x.device)
            is_causal = False
        elif self.bidirectional:
            attn_mask = None
            is_causal = False  # 窗口内双向：无 mask（预测点=窗口末，输入全在过去，无泄露）
        else:
            attn_mask = None
            is_causal = True  # 标准 causal attention（三角 mask 由 flash attention 内部处理）
        if self.flash:
            output = F.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if self.use_alibi:
                scores = scores + attn_mask
            elif not self.bidirectional:
                causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
                scores = scores.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
            output = self.attn_dropout(attn_weights) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = output * torch.sigmoid(self.gate_proj(x))
        return self.resid_dropout(self.o_proj(output))


class FeedForward(nn.Module):
    def __init__(self, config: KLineConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class KLineTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: KLineConfig):
        super().__init__()
        self.self_attn = CausalAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states: torch.Tensor,
                time_positions: torch.Tensor | None = None) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), time_positions)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class KLineFeatureEncoder(nn.Module):
    """v10: raw_proj 仅接收 OHLC 百分比偏移，完全去掉 volume 输入"""

    def __init__(self, hidden_size: int, dropout: float = 0.1, num_symbols: int = 1,
                 close_path: bool = False,
                 use_technical: bool = True,
                 use_vol_oi: bool = False):
        super().__init__()
        self.use_technical = use_technical
        self.close_path = close_path
        self.use_vol_oi = use_vol_oi
        branch_dim = hidden_size // 4  # 每个分支 8 维

        self.raw_proj = nn.Sequential(nn.Linear(4, branch_dim), nn.SiLU(), nn.Dropout(dropout))
        self.kline_feat_mlp = nn.Sequential(
            nn.Linear(5, branch_dim), nn.SiLU(), nn.Dropout(dropout),
        )
        self.temporal_proj = nn.Sequential(
            nn.Linear(9, branch_dim), nn.SiLU(), nn.Dropout(dropout),
        )
        self.technical_proj = nn.Sequential(
            nn.Linear(7, branch_dim), nn.SiLU(), nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.SiLU(), nn.Dropout(dropout),
        )
        # vol/OI 分支：输入3维（量能水平/量比/持仓变化），输出拼进 fusion 前
        if self.use_vol_oi:
            self.vol_oi_proj = nn.Sequential(
                nn.Linear(3, branch_dim), nn.SiLU(), nn.Dropout(dropout),
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_size + branch_dim, hidden_size), nn.SiLU(), nn.Dropout(dropout),
            )
        # 频率标识投影：log2(分钟数) → 加到融合特征上
        self.freq_emb = nn.Embedding(4, hidden_size)
        # 品种嵌入：区分 RB/HC/I 等不同品种
        self.symbol_emb = nn.Embedding(num_symbols, hidden_size)

    def _compute_kline_features(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale = scale.squeeze(-1)  # [B, 1]
        open_p, high, low, close = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        prev_close = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        eps = 1e-8
        hl_range = (high - low).abs() + eps
        # RevIN：收益率和振幅除以窗口波动率，抹平日盘/夜盘及regime间的尺度差异
        log_return = torch.log((close / (prev_close + eps)).clamp(min=eps)) * 100 / scale
        amplitude = (high - low).abs() / (close.abs() + eps) * 100 / scale
        body_ratio = (close - open_p) / hl_range
        upper_shadow = (high - torch.maximum(open_p, close)).clamp(min=0) / hl_range
        lower_shadow = (torch.minimum(open_p, close) - low).clamp(min=0) / hl_range
        return torch.stack([
            log_return, amplitude, body_ratio, upper_shadow, lower_shadow,
        ], dim=-1)

    def _compute_technical_features(self, x: torch.Tensor) -> torch.Tensor:
        close = x[..., 3]
        high = x[..., 1]
        low = x[..., 2]
        eps = 1e-8
        # 因果扩展窗口：窗口前端的 bar 用已有历史算均值，不填充假数据
        ma5 = causal_ma(close, 5)
        ma20 = causal_ma(close, 20)
        ma_diff = (ma20 - ma5).abs() + eps
        ma_position = ((close - ma5) / ma_diff).clamp(-10, 10)  # 横盘时 ma5≈ma20，防止爆炸
        tr = (high - low).abs()
        atr14 = causal_ma(tr, 14) + eps
        vol_regime = tr / atr14
        close_pad_r = F.pad(close.unsqueeze(1), (19, 0), mode='replicate').squeeze(1)
        recent_high = F.max_pool1d(close_pad_r.unsqueeze(1), 20, stride=1).squeeze(1)
        recent_low = -F.max_pool1d((-close_pad_r).unsqueeze(1), 20, stride=1).squeeze(1)
        range_diff = (recent_high - recent_low).abs() + eps
        range_position = (close - recent_low) / range_diff

        prev_close = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        # RSI(14) Cutler 版：SMA 均涨/均跌，输出已映射到 [-1, 1]（超买超卖强度）
        delta = close - prev_close
        avg_gain = causal_ma(delta.clamp(min=0), 14) + eps
        avg_loss = causal_ma((-delta).clamp(min=0), 14) + eps
        rs = avg_gain / avg_loss
        rsi = (100.0 - 100.0 / (1.0 + rs) - 50.0) / 50.0  # warm-up 时 delta=0 → 0.0 中性
        # MACD 代理：ma12-ma26 慢趋势差，以 ATR 定标（趋势强度≈几个ATR）
        ma12 = causal_ma(close, 12)
        ma26 = causal_ma(close, 26)
        macd_norm = ((ma12 - ma26) / atr14).clamp(-5, 5)
        # 布林带宽相对值：20周期σ相对其60周期均值的倍数（波动挤压/扩张状态）
        ma20_sq = causal_ma(close * close, 20)
        std20 = (ma20_sq - ma20 * ma20).clamp(min=0).sqrt() + eps
        boll_squeeze = (std20 / (causal_ma(std20, 60) + eps)).clamp(0, 3)
        # ROC20 动量：20根收益率，用 √20·ATR 定标（方向性动量与噪声之比）
        close_lag20 = F.pad(close, (20, 0), mode='replicate')[:, :-20]
        roc20 = ((close - close_lag20) / (atr14 * 4.4721 + eps)).clamp(-5, 5)

        return torch.stack([
            ma_position, vol_regime, range_position,
            rsi, macd_norm, boll_squeeze, roc20,
        ], dim=-1)

    def forward(self, x, temporal_feat=None, freq_feat=None, symbol_id=None, **kwargs):
        eps = 1e-8
        # RevIN 窗口归一化：以窗口内 bar 间涨跌幅的标准差为尺度
        close_seq = x[..., 3]
        pct_change = (close_seq[:, 1:] - close_seq[:, :-1]) / (close_seq[:, :-1].abs() + eps) * 100
        scale = pct_change.std(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-3)  # [B, 1, 1]
        if self.close_path:
            # 路径锚定：窗口前3根收盘均值为基准（扛单根脏bar），OHLC全列转为
            # "距窗口起点几个常态波动"，携带完整趋势路径；bar内形状由形态通道负责
            anchor = close_seq[:, :3].mean(dim=1, keepdim=True).unsqueeze(-1)  # [B,1,1]
            price_pct = (x[..., :4] - anchor) / (anchor.abs() + eps) * 100 / scale
        else:
            base = x[..., 3:4]  # [B, S, 1] close
            # OHLC 转为相对 close 的百分比点数，再除以窗口波动率
            price_pct = (x[..., :4] - base) / (base.abs() + eps) * 100 / scale  # [B, S, 4]

        raw = self.raw_proj(price_pct)
        kline_feat = self.kline_feat_mlp(self._compute_kline_features(x, scale))
        temporal = self.temporal_proj(temporal_feat) if temporal_feat is not None else torch.zeros_like(raw)
        technical = (
            self.technical_proj(self._compute_technical_features(x))
            if self.use_technical else torch.zeros_like(raw)
        )
        fused_parts = [raw, kline_feat, temporal, technical]
        if self.use_vol_oi:
            # x[...,4]=log1p(volume) z-scored, x[...,5]=log1p(close_oi) z-scored（dataset 侧已处理）
            vol = x[..., 4]
            oi = x[..., 5]
            vol_ma = causal_ma(vol, 20)
            vol_ratio = vol - vol_ma                      # 量比（log域差=相对20均量的倍数）
            oi_prev = torch.cat([oi[:, :1], oi[:, :-1]], dim=1)
            oi_delta = oi - oi_prev                       # 持仓变化（log域≈变化率）
            oi_delta = torch.cat([oi_delta[:, :1] * 0, oi_delta[:, 1:]], dim=1)  # 首bar置零
            fused_parts.append(self.vol_oi_proj(
                torch.stack([vol, vol_ratio, oi_delta], dim=-1).clamp(-10, 10)
            ))
        fused = self.fusion(torch.cat(fused_parts, dim=-1))
        if freq_feat is not None:
            # [B] → [B, 1, H]，广播到所有时间步
            fused = fused + self.freq_emb(freq_feat.long()).unsqueeze(1)
        if symbol_id is not None:
            # [B] → [B, 1, H]，广播到所有时间步
            fused = fused + self.symbol_emb(symbol_id).unsqueeze(1)
        return fused


class DailyContextEncoder(nn.Module):
    """日线上下文编码器：每根日K编码成形状token，作为分钟序列的前缀。

    与分钟bar同构的5维形状特征（对数收益/振幅/实体比/上下影），
    RevIN 式定标：除以本段日线窗口的收益波动率。"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden_size), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
        )

    def forward(self, daily: torch.Tensor) -> torch.Tensor:
        # daily: [B, D, 4] 原始 OHLC（已保证因果：只含已收盘日K）
        eps = 1e-8
        open_p, high, low, close = daily[..., 0], daily[..., 1], daily[..., 2], daily[..., 3]
        prev_close = torch.cat([close[:, :1], close[:, :-1]], dim=1)
        log_ret = torch.log((close / (prev_close + eps)).clamp(min=eps))
        scale = log_ret.std(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1] 日收益波动率
        hl_range = (high - low).abs() + eps
        feats = torch.stack([
            log_ret * 100 / scale,
            (high - low).abs() / (close.abs() + eps) * 100 / scale,
            (close - open_p) / hl_range,
            (high - torch.maximum(open_p, close)).clamp(min=0) / hl_range,
            (torch.minimum(open_p, close) - low).clamp(min=0) / hl_range,
        ], dim=-1)  # [B, D, 5]
        return self.mlp(feats)  # [B, D, H]


class ForeignContextEncoder(nn.Module):
    """外盘日线编码器：收盘-only 序列（Wind EDB 无 OHLC），3 维特征
    （日对数收益 / 5日动量 / 20日区间位置，dataset 侧已算好）→ token。
    收益/动量按本段窗口波动率定标，与内盘 RevIN 同构。"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), nn.SiLU(),
        )

    def forward(self, fctx: torch.Tensor) -> torch.Tensor:
        # fctx: [B, D2, 3]（log_ret, mom5, pos20）
        ret, mom, pos = fctx[..., 0], fctx[..., 1], fctx[..., 2]
        scale = ret.std(dim=1, keepdim=True).clamp(min=1e-6)  # [B,1]
        feats = torch.stack([ret / scale, mom / scale, pos * 2 - 1], dim=-1)  # pos20→[-1,1]
        return self.mlp(feats)  # [B, D2, H]


class CrossSymbolEncoder(nn.Module):
    """跨品种 variate token 编码器（iTransformer 启发）。

    每个品种的 N 根已收盘日K → DailyContextEncoder 逐日编码 → 取最后位置池化
    成 1 个"品种 token"（variate token）；13 个品种 token 间做一层**双向**
    self-attention（全部输入都是过去的日K，无未来泄露），捕获板块内联动
    （黑色系 rb/hc/i/j/jm 共振、油脂 p/y/m 共振等）。
    输出 [B, NS, H] 作为前缀 token 拼进主干（causal 主干中分钟 token 可 attend）。
    缺失品种由 cross_mask 屏蔽（attention 里 -inf）。
    """

    def __init__(self, hidden_size: int, num_symbols: int, dropout: float = 0.1):
        super().__init__()
        self.daily_enc = DailyContextEncoder(hidden_size, dropout=dropout)
        self.sym_emb = nn.Embedding(num_symbols, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=max(1, hidden_size // 16),
            dim_feedforward=hidden_size * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.cross_attn = nn.TransformerEncoder(layer, num_layers=1)
        self.out_norm = RMSNorm(hidden_size)

    def forward(self, cross_ctx: torch.Tensor, cross_mask: torch.Tensor,
                symbol_ids: torch.Tensor) -> torch.Tensor:
        # cross_ctx: [B, NS, D, 4] 原始OHLC；cross_mask: [B, NS]（1=有数据）；
        # symbol_ids: [NS] 品种id表（与 dataset 侧 _cross_ids 同序）
        B, NS, D, _ = cross_ctx.shape
        if NS != symbol_ids.numel():
            raise ValueError(
                f"cross_ctx 品种数({NS})与模型 cross_symbol_ids({symbol_ids.numel()})不一致；"
                f"数据集构造时必须用训练时的品种全集"
            )
        tok = self.daily_enc(cross_ctx.reshape(B * NS, D, 4))  # [B*NS, D, H]
        vt = tok[:, -1, :].reshape(B, NS, -1)                  # 最后位置池化
        vt = vt + self.sym_emb(symbol_ids).unsqueeze(0)        # 品种身份
        pad = (cross_mask < 0.5)                               # True=屏蔽
        vt = self.cross_attn(vt, src_key_padding_mask=pad)
        return self.out_norm(vt)  # [B, NS, H]


class SectorTokenEncoder(nn.Module):
    """板块分组 token 编码器（iTransformer variate token 的轻量降级版）。

    与 CrossSymbolEncoder 的唯一区别：组内**带 mask 均值**替代跨品种 attention。
    不引入新的时序/注意力参数（日K编码复用主干已有的 DailyContextEncoder），
    仅一张组 embedding 表——cross 配方 8 倍噪带的种子方差来源（过富参数）被切除，
    保留"品种能看到所属板块整体温度"的核心语义。
    """

    def __init__(self, hidden_size: int, n_groups: int):
        super().__init__()
        self.grp_emb = nn.Embedding(n_groups, hidden_size)

    def forward(self, cross_ctx: torch.Tensor, cross_mask: torch.Tensor,
                cross_ids: torch.Tensor, groups: list[list[int]],
                daily_enc: DailyContextEncoder) -> torch.Tensor:
        # cross_ctx: [B, NS, D, 4]；cross_mask: [B, NS]；cross_ids: [NS] 与 cross_ctx 同序
        B, NS, D, _ = cross_ctx.shape
        tok = daily_enc(cross_ctx.reshape(B * NS, D, 4))
        vt = tok[:, -1, :].reshape(B, NS, -1)          # 每品种1个variate向量
        outs = []
        for gid, members in enumerate(groups):
            # members 为品种id → 转成 cross_ids 中的位置
            pos = [(cross_ids == m).nonzero(as_tuple=True)[0] for m in members]
            pos = [p for p in pos if p.numel() > 0]
            if not pos:
                continue
            idx = torch.cat(pos)
            w = cross_mask[:, idx]                     # [B, n_mem]
            g = (vt[:, idx] * w.unsqueeze(-1)).sum(1) / w.sum(1, keepdim=True).clamp(min=1e-8)
            outs.append(g + self.grp_emb.weight[gid])
        return torch.stack(outs, dim=1)                # [B, G, H]


class KLineTransformer(nn.Module):
    def __init__(self, config: KLineConfig | None = None):
        super().__init__()
        self.config = config or KLineConfig()
        cfg = self.config

        self.input_encoder = KLineFeatureEncoder(
            cfg.hidden_size, dropout=cfg.dropout, num_symbols=cfg.num_symbols,
            use_technical=cfg.use_technical, close_path=getattr(cfg, "close_path", False),
            use_vol_oi=getattr(cfg, "use_vol_oi", False),
        )
        self.dropout = nn.Dropout(cfg.dropout)
        # Conv stem：causal 1D 卷积，在进 attention 前捕获相邻 K 线的局部形态
        self.conv_stem = nn.Conv1d(cfg.hidden_size, cfg.hidden_size, kernel_size=3)
        self.layers = nn.ModuleList([KLineTransformerBlock(l, cfg) for l in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=config.rms_norm_eps)
        # 可学习位置编码（标准 Transformer，无 ALiBi 时使用）
        self.pos_emb = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)

        self.close_head = nn.Linear(cfg.hidden_size, 1)
        # 日线金字塔支路（daily_bars>0 时启用）
        self.daily_encoder = (
            DailyContextEncoder(cfg.hidden_size, dropout=cfg.dropout)
            if cfg.daily_bars > 0 else None
        )
        # 外盘日线支路（foreign_bars>0 时启用）+ 段嵌入（0=内盘日K, 1=外盘日K）
        self.foreign_encoder = (
            ForeignContextEncoder(cfg.hidden_size, dropout=cfg.dropout)
            if cfg.foreign_bars > 0 else None
        )
        if self.foreign_encoder is not None:
            self.seg_emb = nn.Embedding(2, cfg.hidden_size)
        # 跨品种 variate token 支路（cross_daily_bars>0 且未启用 sector 版时启用；
        # sector_groups 非空时 cross_ctx 由 sector_encoder 消费，两者互斥）
        self.cross_encoder = (
            CrossSymbolEncoder(cfg.hidden_size, cfg.num_symbols, dropout=cfg.dropout)
            if getattr(cfg, "cross_daily_bars", 0) > 0 and not getattr(cfg, "sector_groups", None)
            else None
        )
        if self.cross_encoder is not None:
            ids = cfg.cross_symbol_ids or list(range(cfg.num_symbols))
            self.register_buffer("cross_ids", torch.tensor(ids, dtype=torch.long))
        # 板块分组 token 支路（sector_groups 非空时启用；与 cross_encoder 互斥，
        # 同为 cross_ctx 的两种消费方式，sector 为零时序参数的轻量版）
        self.sector_encoder = None
        self.sector_groups = getattr(cfg, "sector_groups", None)
        if self.sector_groups:
            if self.daily_encoder is None:
                raise ValueError("sector_groups 需要 daily_bars>0（复用 DailyContextEncoder）")
            self.sector_encoder = SectorTokenEncoder(cfg.hidden_size, len(self.sector_groups))
            ids = cfg.cross_symbol_ids or list(range(cfg.num_symbols))
            self.register_buffer("cross_ids", torch.tensor(ids, dtype=torch.long))
        # 细粒度上下文支路（fine_bars>0 时启用）：与日K编码器同构（同 5 维形状特征），
        # 独立参数 + 独立段嵌入，拼在分钟主窗口紧前方（微观结构紧邻预测点）
        self.fine_encoder = (
            DailyContextEncoder(cfg.hidden_size, dropout=cfg.dropout)
            if getattr(cfg, "fine_bars", 0) > 0 else None
        )
        if self.fine_encoder is not None:
            self.fine_seg_emb = nn.Embedding(1, cfg.hidden_size)
        if cfg.task == "classify":
            self.class_head = nn.Linear(cfg.hidden_size, cfg.num_classes)
            w = torch.tensor(cfg.class_weights, dtype=torch.float32) if cfg.class_weights else None
            self.register_buffer("ce_weights", w)
            # E3 路径状态辅助头：pooled 共享表示 + 节点 embedding → 4 节点 × 3 态
            # （教师模型建议：4 个节点含义不同，需要 node embedding 区分，不能共用一个裸线性层）
            if getattr(cfg, "path_aux", False):
                if getattr(cfg, "query_decoder", False):
                    # E6'：未来时间 query decoder —— 4 个可学习 query 代表未来 25/50/75/100%，
                    # cross-attend 全部历史 encoder 输出（历史皆合法，无未来泄漏），
                    # 残差 + LayerNorm 后逐节点出 3 态；替代 pooled+node_emb 简易头
                    self.path_queries = nn.Parameter(torch.zeros(4, cfg.hidden_size))
                    nn.init.normal_(self.path_queries, std=0.02)
                    self.path_cross_attn = nn.MultiheadAttention(
                        cfg.hidden_size, cfg.num_attention_heads, batch_first=True)
                    self.path_query_norm = nn.LayerNorm(cfg.hidden_size)
                    self.path_head = nn.Linear(cfg.hidden_size, 3)
                else:
                    self.path_node_emb = nn.Embedding(4, cfg.hidden_size)
                    self.path_head = nn.Linear(cfg.hidden_size, 3)
            # E4 excursion 分桶头：pooled → 下行/上行最大偏移各占 θ 的几分（6 桶）
            if getattr(cfg, "exc_aux", False):
                self.exc_head_dn = nn.Linear(cfg.hidden_size, 6)
                self.exc_head_up = nn.Linear(cfg.hidden_size, 6)
            # ViT 式读出：last=最后位置（单向标配）；mean=全局平均；cls=可学习 [CLS] token
            self.readout = getattr(cfg, "readout", "last")
            if self.readout == "cls":
                self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.hidden_size))
                nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, kline_seq, targets=None, temporal_feat=None, hourly_ctx=None, time_pos=None, freq_feat=None, symbol_id=None, labels=None, daily_ctx=None, foreign_ctx=None, soft_labels=None, cross_ctx=None, cross_mask=None, fine_ctx=None):
        bsz, seq_len, _ = kline_seq.shape
        x = self.input_encoder(kline_seq, temporal_feat=temporal_feat, freq_feat=freq_feat, symbol_id=symbol_id)
        # 日线金字塔：完整日K编码为前缀token，拼在分钟序列前（因果注意力下分钟token可 attend 日线背景）
        prefix = []
        if self.daily_encoder is not None and daily_ctx is not None:
            dtok = self.daily_encoder(daily_ctx)  # [B, D, H]
            if self.foreign_encoder is not None:
                dtok = dtok + self.seg_emb.weight[0]
            prefix.append(dtok)
        if self.foreign_encoder is not None and foreign_ctx is not None:
            ftok = self.foreign_encoder(foreign_ctx) + self.seg_emb.weight[1]  # [B, D2, H]
            prefix.append(ftok)
        if self.cross_encoder is not None and cross_ctx is not None:
            # 跨品种 variate token：板块联动信息，拼在外盘之后、分钟序列之前
            prefix.append(self.cross_encoder(cross_ctx, cross_mask, self.cross_ids))
        if self.sector_encoder is not None and cross_ctx is not None:
            # 板块分组 token：轻量版联动信息（组内均值，无跨品种attention）
            prefix.append(self.sector_encoder(cross_ctx, cross_mask, self.cross_ids,
                                              self.sector_groups, self.daily_encoder))
        if self.fine_encoder is not None and fine_ctx is not None:
            # 细粒度 token：最近 N 根已收盘 5/15m bar 的微观结构，紧邻分钟主窗口
            prefix.append(self.fine_encoder(fine_ctx) + self.fine_seg_emb.weight[0])
        if prefix:
            x = torch.cat(prefix + [x], dim=1)
        # [CLS] token 插到序列最前（在 pos_emb 之前，拿位置0的编码；
        # conv_stem 左侧 replicate 填充，首位只卷到自己，无信息污染）
        use_cls = getattr(self, "readout", "last") == "cls" and self.config.task == "classify"
        if use_cls:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.dropout(x)
        # 可学习位置编码
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)
        # 时间感知 RoPE 的整序列位置：分钟段用 time_pos（真实流逝时间），
        # 前缀 token 按 1 bar 单位均匀排到窗口起点之前（语义粒度不同，间距无关大局）
        rope_positions = None
        if any(getattr(l.self_attn, "rope", False) for l in self.layers) and time_pos is not None:
            n_prefix = x.shape[1] - time_pos.shape[1]
            if n_prefix > 0:
                offs = torch.arange(n_prefix, 0, -1, device=x.device, dtype=time_pos.dtype)
                prefix_pos = time_pos[:, :1] - offs.unsqueeze(0)  # [B, n_prefix]（含 CLS 位）
                rope_positions = torch.cat([prefix_pos, time_pos], dim=1)
            else:
                rope_positions = time_pos
        # causal padding：左侧 replicate 填充 2 步，卷积后长度不变
        x = self.conv_stem(F.pad(x.transpose(1, 2), (2, 0), mode='replicate')).transpose(1, 2)
        for layer in self.layers:
            x = layer(x, rope_positions)
        x = self.norm(x)

        if self.config.task == "classify":
            if use_cls:
                pooled = x[:, 0, :]          # ViT 式：[CLS] token 读出
            elif getattr(self, "readout", "last") == "mean":
                pooled = x.mean(dim=1)       # 全局平均投票
            else:
                pooled = x[:, -1, :]         # 现状：最后位置（单向标配）
            logits = self.class_head(pooled)  # [B, 3]
            result = {"logits": logits, "pred_class": logits.argmax(dim=-1)}
            if getattr(self, "path_head", None) is not None:
                # 损失在 trainer 端 masked CE，-1 节点忽略
                if getattr(self.config, "query_decoder", False):
                    # E6'：query 提取"这段历史对未来各时点的含义"
                    q = self.path_queries.unsqueeze(0).expand(x.shape[0], -1, -1)  # [B,4,H]
                    a, _ = self.path_cross_attn(q, x, x)  # query=未来时点, kv=历史
                    h_i = self.path_query_norm(q + a)     # 残差 + LN
                    result["path_logits"] = self.path_head(h_i)  # [B,4,3]
                else:
                    # E3：pooled + 各节点 embedding → 逐节点 3 态 logits [B, 4, 3]
                    h_i = pooled.unsqueeze(1) + self.path_node_emb.weight.unsqueeze(0)  # [B,4,H]
                    result["path_logits"] = self.path_head(h_i)  # [B,4,3]
            if getattr(self, "exc_head_dn", None) is not None:
                # E4：excursion 分桶 logits，各 [B, 6]
                result["exc_logits_dn"] = self.exc_head_dn(pooled)
                result["exc_logits_up"] = self.exc_head_up(pooled)
            if labels is not None:
                if soft_labels is not None:
                    # 软标签 v2：混合损失 = 0.5×硬CE（带类别权重，保底 argmax 语义）
                    #           + 0.5×软CE（不带类别权重——软目标本身已是逐样本分布，
                    #             再乘方向类权重会怂恿模型永远选方向，v1 的坑）
                    logp = F.log_softmax(logits, dim=-1)
                    soft_loss = -(soft_labels * logp).sum(-1).mean()
                    hard_loss = F.cross_entropy(logits, labels, weight=self.ce_weights)
                    result["loss"] = 0.5 * hard_loss + 0.5 * soft_loss
                else:
                    result["loss"] = F.cross_entropy(logits, labels, weight=self.ce_weights)
                # 方向间隔损失：真实有方向的样本，要求赢家 logit 领先输家 ≥ margin，
                # 惩罚"多≈空"的骑墙输出；软标签的方向证据强度 |t多−t空| 做权重调制，
                # 冲到 92% 的样本要求大分离，纯震荡样本不罚骑墙；"无"样本不参与
                mw = getattr(self.config, "margin_weight", 0.0)
                if mw > 0:
                    sep = logits[:, 2] - logits[:, 0]  # 多−空 的 logit 间隔
                    sign = (labels == 2).float() - (labels == 0).float()  # 正→+1 负→−1
                    active = (labels != 1).float()
                    if soft_labels is not None:
                        w = active * (soft_labels[:, 2] - soft_labels[:, 0]).abs().detach()
                    else:
                        w = active
                    m_loss = (F.relu(self.config.margin - sep * sign) * w).sum() / w.sum().clamp(min=1e-8)
                    result["loss"] = result["loss"] + mw * m_loss
                    result["loss_margin"] = m_loss
                # 密集辅助监督：逐 bar 预测下一根收益，强制编码器学习路径动力学
                # （语音式逐帧监督；dense_tail 只监督窗口末尾上下文充足的位置）
                if targets is not None and self.config.dense_loss_weight > 0:
                    tail = self.config.dense_tail
                    all_logits = self.close_head(x)  # [B, S, 1]
                    logits_tail = all_logits[:, -tail:, 0].contiguous()
                    true_pct_tail = ((targets[:, -tail:, 3] - kline_seq[:, -tail:, 3]) / \
                        (kline_seq[:, -tail:, 3].abs() + 1e-8) * 100).contiguous()
                    loss_aux = F.smooth_l1_loss(logits_tail, true_pct_tail,
                                                beta=self.config.huber_delta)
                    result["loss"] = result["loss"] + self.config.dense_loss_weight * loss_aux
                    result["loss_aux"] = loss_aux
            else:
                result["loss"] = logits.sum() * 0.0
            return result

        all_logits = self.close_head(x)  # [B, S, 1] 每个位置的预测
        last_hidden = x[:, -1, :]
        close_pct = all_logits[:, -1, 0]  # [B] 最后一根K线的预测

        # 只构造最后一个时间步的预测值
        base = kline_seq[..., 3]  # [B, S]
        pred_close = base.clone()
        pred_close[:, -1] = base[:, -1] * (1 + close_pct / 100)

        open_pred = base
        high_pred = torch.maximum(open_pred, pred_close) + 0.01
        low_pred = torch.minimum(open_pred, pred_close) - 0.01

        pred = torch.stack([open_pred, high_pred, low_pred, pred_close], dim=-1)
        result = {"pred": pred, "close_pct": close_pct}

        if targets is not None:
            true_base = kline_seq[:, -1, 3]
            # targets 序列的最后一个元素 = 输入窗口末尾之后第 target_offset 根 K 线的 close
            true_close = targets[:, -1, 3]
            true_close_pct = (true_close - true_base) / (true_base.abs() + 1e-8) * 100
            if self.config.loss_type == "huber":
                loss_last = F.smooth_l1_loss(close_pct, true_close_pct, beta=self.config.huber_delta)
            else:
                loss_last = F.mse_loss(close_pct, true_close_pct)

            loss = loss_last
            if self.config.dense_loss_weight > 0:
                # 密集监督：窗口末尾 dense_tail 个位置 t 预测 t+offset 的涨跌幅
                # targets[:, t, 3] = 位置 t 之后第 offset 根K线的 close
                # contiguous()：MPS 后端对非连续切片的 smooth_l1_loss 会报 view size 错误
                tail = self.config.dense_tail
                logits_tail = all_logits[:, -tail:, 0].contiguous()
                true_pct_tail = ((targets[:, -tail:, 3] - kline_seq[:, -tail:, 3]) / \
                    (kline_seq[:, -tail:, 3].abs() + 1e-8) * 100).contiguous()
                if self.config.loss_type == "huber":
                    loss_all = F.smooth_l1_loss(logits_tail, true_pct_tail, beta=self.config.huber_delta)
                else:
                    loss_all = F.mse_loss(logits_tail, true_pct_tail)
                loss = loss + self.config.dense_loss_weight * loss_all
            result["loss"] = loss

        return result

    @torch.no_grad()
    def predict_next(self, kline_seq: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self.forward(kline_seq)
        return out["pred"][:, -1, :]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str) -> None:
        torch.save({"model": self.state_dict(), "config": self.config}, path)

    def load_checkpoint(self, path: str, device: str = "cpu") -> None:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.load_state_dict(ckpt["model"])
        print(f"已加载模型权重: {path}")
