"""Identity depth growth with bounded, explicit exposure and matmul budgets."""

import math

import torch
from torch import nn

from . import history_structure as structure

DEPTHS = (4, 8, 12)
LR_SCALES = (1, 3)


def install(model, depth, seed):
    backbone = model.core.encoder.backbone
    old = list(backbone.layers)
    if len(old) != 4 or depth not in DEPTHS or hasattr(model.core.encoder, "pool"):
        raise ValueError("Unpooled four-layer source and depth4/8/12 required")
    first = old[0]
    width, ff = first.self_attn.embed_dim, first.linear1.out_features
    if any(
        not layer.norm_first or layer.self_attn.num_heads != 8 or layer.dropout.p != 0
        for layer in old
    ):
        raise ValueError("Original prenorm eight-head dropout0 backbone required")
    # Layers4..7 have identical initialization in depth8 and depth12. No RNG leak.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 20260929)
        for _ in range(depth - 4):
            layer = nn.TransformerEncoderLayer(
                width, 8, ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            ).to(device=first.linear1.weight.device, dtype=first.linear1.weight.dtype)
            with torch.no_grad():
                layer.self_attn.out_proj.weight.zero_()
                layer.self_attn.out_proj.bias.zero_()
                layer.linear2.weight.zero_()
                layer.linear2.bias.zero_()
            old.append(layer)
    backbone.layers = nn.ModuleList(old)
    return model


def matmul_proxy(depth):
    """Approximate training MAC/view, not measured GPU time or exact FLOPs.

    Includes dense projections, full attention products, five127-age query heads,
    four local reads and a factor3 for forward+backward. Excludes norm/softmax,
    activation/optimizer/data/validation and kernel implementation differences.
    Input budget includes both original28 and path1 projections conservatively.
    """
    if depth not in DEPTHS:
        raise ValueError("Undeclared depth")
    t, w, ff = 128, 768, 3072
    layer = 4 * t * w * w + 2 * t * t * w + 2 * t * w * ff
    encoder = depth * layer + t * 29 * w + t * w * w
    q, d, memory, qff = 127, 192, 4, 384
    block = 2 * q * d * d + 2 * memory * d * d + 2 * q * memory * d + 2 * q * d * qff
    query = 5 * (w * w + q * d * d + 2 * block + q * d * 7)
    local = 4 * (w * 256 + 256 * 16 * 7)
    return int(3 * (encoder + query + local))


def budgets(epochs=100):
    if epochs <= 5:
        raise ValueError("Base budget must exceed warmup")
    c4 = matmul_proxy(4)
    c8 = math.ceil(epochs * matmul_proxy(8) / c4)
    c12 = math.ceil(epochs * matmul_proxy(12) / c4)
    return {"d4": epochs, "d8": epochs, "d12": epochs, "d4_c8": c8, "d4_c12": c12}


def stages(meta, depth):
    b = meta["budgets"]
    return [b["d4"], b["d4_c8"], b["d4_c12"]] if depth == 4 else [b[f"d{depth}"]]


def gradient_groups(model, query):
    return [
        list(model.core.encoder.parameters()),
        list(model.local_head.parameters()),
        list(query.parameters()),
    ]


def losses(model, query, batch, prefixes, query_prefixes, shifts, statistics, local, mode):
    if mode not in ("d4", "d8", "d12"):
        raise ValueError("Unknown depth arm")
    return structure.losses(
        model, query, batch, prefixes, query_prefixes, shifts, statistics, local, "structure"
    )


metrics = structure.metrics
