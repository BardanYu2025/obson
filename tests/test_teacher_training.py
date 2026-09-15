"""Teacher 训练线的无 GPU 回归测试。

覆盖不依赖 torch 的标签派生语义，并检查默认配置不会打开 Teacher 头。
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_teacher_is_opt_in_in_config_source():
    source = (ROOT / "src/obson/model/transformer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "utility_head":
            fields.append(node.value)
    assert len(fields) == 1
    assert isinstance(fields[0], ast.Constant) and fields[0].value is False


def test_teacher_cli_is_explicitly_opt_in():
    source = (ROOT / "scripts/train_multi_symbol.py").read_text(encoding="utf-8")
    assert '"--teacher"' in source
    assert "utility_head=args.teacher" in source
    assert "utility_logit_weight=args.teacher_logit_weight if args.teacher else 0.0" in source


def test_utility_target_semantics_are_production_aligned():
    source = (ROOT / "src/obson/model/dataset.py").read_text(encoding="utf-8")
    # The target must encode [short, long], with hit +0.8 and opposite -0.5.
    assert "short_u = 0.8 if label == 0 else (-0.5 if label == 2 else -fwd_theta)" in source
    assert "long_u = 0.8 if label == 2 else (-0.5 if label == 0 else fwd_theta)" in source
    assert 'item["utility_target"]' in source


def test_legacy_path_does_not_use_teacher_score():
    source = (ROOT / "src/obson/model/mixed_trainer.py").read_text(encoding="utf-8")
    # Composite Teacher selection must be guarded by the opt-in config flag.
    assert 'if getattr(self.model.config, "utility_head", False):' in source

