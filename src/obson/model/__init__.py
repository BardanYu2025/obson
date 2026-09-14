"""obson.model — K-Line Transformer 预测模型

移植自 minimind (https://github.com/jingyaogong/minimind)，
将 decode-only 语言模型改造为连续值时间序列预测器。
"""

from obson.model.dataset import KLineDataset, build_datasets, weekly_seq_len
from obson.model.mixed_trainer import MixedFrequencyTrainer
from obson.model.trainer import KLineTrainer
from obson.model.transformer import KLineConfig, KLineTransformer

__all__ = [
    "KLineConfig",
    "KLineTransformer",
    "KLineDataset",
    "build_datasets",
    "weekly_seq_len",
    "KLineTrainer",
    "MixedFrequencyTrainer",
]
