"""Comparable rule/path/model retrieval with frozen transforms and as-of outcomes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import SCHEMA
from .data import manifest, split_boundaries
from .metrics import block_interval
from .structure import describe, descriptor


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def path_vector(s, row, window):
    close = s.frame.close.to_numpy()[row - window + 1 : row + 1]
    return np.interp(
        np.linspace(0, len(close) - 1, 32),
        np.arange(len(close)),
        (close - close[-1]) / s.labels["atr"][row],
    ).astype(np.float32)


def save_index(series, path, window=128, stride=16, checkpoint=None, device="cpu"):
    if window < 16 or stride < 1:
        raise ValueError("window must be >=16 and stride positive")
    rows = [
        (i, j)
        for i, s in enumerate(series)
        for j in range(window - 1, len(s.frame), stride)
        if s.main[j]
    ]
    if not rows:
        raise ValueError("No eligible index windows")
    sid, endpoint = np.array(rows, np.int64).T
    rules = np.stack([descriptor(series[i].labels, j) for i, j in rows])
    paths = np.stack([path_vector(series[i], j, window) for i, j in rows])
    meta = {
        "schema": SCHEMA,
        "window": window,
        "stride": stride,
        "manifest": manifest(series),
        "checkpoint": None,
        "model_available_after": None,
        "feature_transform": "fixed rules/path; model train-only centering",
    }
    arrays = {"series": sid, "row": endpoint, "rule": normalize(rules), "path": normalize(paths)}
    if checkpoint:
        import torch

        from .model import checkpoint_hash, load_model

        model, ck = load_model(checkpoint, device)
        if window not in model.cfg.trained_windows:
            raise ValueError(
                f"Index window must be one of trained lengths {model.cfg.trained_windows}"
            )
        vectors = []
        with torch.no_grad():
            for lo in range(0, len(rows), 128):
                x = np.stack([series[i].x[j - window + 1 : j + 1] for i, j in rows[lo : lo + 128]])
                vectors.append(model(torch.from_numpy(x).to(device))["embedding"].cpu().numpy())
        vectors = np.concatenate(vectors)
        bound = np.datetime64(ck["manifest"]["boundaries"]["train_until"])
        train = np.array([series[i].sessions[j] <= bound for i, j in rows])
        if not train.any():
            raise ValueError("No training-period samples available for frozen centering")
        mean = vectors[train].mean(0)
        arrays.update(model=normalize(vectors - mean), model_mean=mean)
        meta["checkpoint"] = checkpoint_hash(checkpoint)
        # Validation chose the checkpoint, so historical deployment starts AFTER validation.
        meta["model_available_after"] = ck["manifest"]["boundaries"]["val_until"]
    arrays["metadata"] = np.array(json.dumps(meta))
    path = Path(path)
    if path.exists():
        raise ValueError("Index already exists; use a new filename for reproducibility")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    return {
        "path": str(path),
        "entries": len(rows),
        "methods": [k for k in ("rule", "path", "model") if k in arrays],
    }


class Engine:
    def __init__(self, series, index, checkpoint=None, diagnostics=()):
        self.series = series
        self.diagnostics = list(diagnostics)
        with np.load(index, allow_pickle=False) as z:
            self.arrays = {k: z[k] for k in z.files}
        self.meta = json.loads(str(self.arrays["metadata"]))
        if (
            self.meta["schema"] != SCHEMA
            or self.meta["manifest"]["sources"] != manifest(series)["sources"]
        ):
            raise ValueError(
                "Index/data mismatch: rebuild the index after changing data, symbols or periods"
            )
        self.window = self.meta["window"]
        self.sid, self.rows = self.arrays["series"], self.arrays["row"]
        self.end = np.array([series[i].ends[j] for i, j in zip(self.sid, self.rows, strict=True)])
        self.start = np.array(
            [
                series[i].frame.datetime.iloc[j - self.window + 1].to_datetime64()
                for i, j in zip(self.sid, self.rows, strict=True)
            ]
        )
        self.model = None
        if checkpoint:
            from .model import checkpoint_hash, load_model

            if checkpoint_hash(checkpoint) != self.meta["checkpoint"]:
                raise ValueError("Checkpoint/index mismatch")
            self.model, _ = load_model(checkpoint)
        self.methods = [
            k
            for k in ("rule", "path", "model")
            if k in self.arrays and (k != "model" or self.model)
        ]

    def locate(self, code, period, asof=None):
        cutoff = (
            np.datetime64(pd.Timestamp(asof).to_datetime64())
            if asof
            else np.datetime64("now", "ns") + np.timedelta64(8, "h")
        )
        candidates = []
        for i, s in enumerate(self.series):
            if s.code != code or s.period != period:
                continue
            ok = np.flatnonzero(
                s.main & (s.ends <= cutoff) & (np.arange(len(s.frame)) >= self.window - 1)
            )
            if len(ok):
                candidates.append((s.ends[ok[-1]], i, int(ok[-1])))
        if not candidates:
            raise ValueError("No completed, eligible contract window at that time")
        _, i, row = max(candidates)
        return i, row

    def vector(self, i, row, method):
        s = self.series[i]
        if method == "rule":
            return normalize(descriptor(s.labels, row))
        if method == "path":
            return normalize(path_vector(s, row, self.window))
        if method != "model" or self.model is None:
            raise ValueError("Model retrieval requires a matching Babel checkpoint and index")
        if s.sessions[row] <= np.datetime64(self.meta["model_available_after"]):
            raise ValueError(
                "Model was selected after this historical time; only post-validation queries are legal"
            )
        import torch

        with torch.no_grad():
            out = self.model(torch.from_numpy(s.x[row - self.window + 1 : row + 1]).unsqueeze(0))
        return normalize(out["embedding"][0].numpy() - self.arrays["model_mean"])

    def retrieve(self, i, row, method="rule", topk=8):
        if not 1 <= topk <= 50:
            raise ValueError("topk must be between 1 and 50")
        s = self.series[i]
        # Physical-time embargo across ALL symbols and frequencies: the candidate
        # must finish before this query window begins. Removes near-duplicate episodes.
        qstart = s.frame.datetime.iloc[row - self.window + 1].to_datetime64()
        eligible = self.end < qstart
        q = self.vector(i, row, method)
        score = self.arrays[method] @ q
        order = np.flatnonzero(eligible)
        order = order[np.argsort(-score[order], kind="stable")]
        hits, intervals = [], []
        for j in order:
            # Hits also cannot repeat the same period of market history, even in
            # another correlated contract. Fewer than K is reported honestly.
            if any(self.start[j] <= hi and self.end[j] >= lo for lo, hi in intervals):
                continue
            hits.append((int(j), float(score[j])))
            intervals.append((self.start[j], self.end[j]))
            if len(hits) == topk:
                break
        return hits

    def snapshot(self, i, row, asof=None, include_future=False):
        s = self.series[i]
        lo = row - self.window + 1
        cutoff = np.datetime64(asof) if asof is not None else s.ends[row]
        tail = min(row + 20, len(s.frame) - 1) if include_future else row
        while tail > row and s.ends[tail] > cutoff:
            tail -= 1
        f = s.frame.iloc[lo : tail + 1]
        bars = [
            {
                "time": str(t),
                "open": float(o),
                "high": float(h),
                "low": float(low),
                "close": float(c),
                "volume": float(v),
                "oi": float(oi),
                "row": lo + k,
            }
            for k, (t, o, h, low, c, v, oi) in enumerate(
                f[["datetime", "open", "high", "low", "close", "volume", "oi"]].itertuples(
                    index=False, name=None
                )
            )
        ]
        outcomes = {}
        for horizon in (5, 10, 20):
            target = row + horizon
            available = target < len(s.frame) and s.ends[target] <= cutoff
            outcomes[str(horizon)] = (
                float((s.frame.close.iloc[target] - s.frame.close.iloc[row]) / s.labels["atr"][row])
                if available
                else None
            )
        return {
            "code": s.code,
            "period": s.period,
            "contract": s.contract,
            "row": row,
            "available_at": str(s.ends[row]),
            "session": str(s.sessions[row]),
            "bars": bars,
            "structure": describe(s.frame, s.labels, row),
            "outcomes": outcomes,
            "oi_available": bool(s.frame.oi_available.iloc[row]),
        }

    def query(self, code, period, asof=None, method="rule", topk=8):
        i, row = self.locate(code, period, asof)
        s = self.series[i]
        hits = self.retrieve(i, row, method, topk)
        query = self.snapshot(i, row)
        matches = []
        for idx, score in hits:
            match = self.snapshot(
                int(self.sid[idx]), int(self.rows[idx]), s.ends[row], include_future=True
            )
            match["similarity"] = score
            matches.append(match)
        estimates = None
        if self.model and s.sessions[row] > np.datetime64(self.meta["model_available_after"]):
            import torch

            with torch.no_grad():
                out = self.model(
                    torch.from_numpy(s.x[row - self.window + 1 : row + 1]).unsqueeze(0)
                )
            estimates = {
                k: out[k][0, -1].softmax(-1).numpy().tolist()
                for k in ("direction", "state", "event")
            }
        return {
            "schema": SCHEMA,
            "method": method,
            "requested_asof": asof,
            "query": query,
            "matches": matches,
            "model_estimates": estimates,
            "model_status": "experimental rule-supervised representation; transfer not certified",
            "notes": self.diagnostics
            + [
                "未确认拐点不会回填；支撑压力与形态均为规则证据，不是交易指令。",
                "相似案例按时间去重；后续收益仅显示查询时刻已发生的部分。",
                "不同周期的后N根不代表相同时间跨度；不混合计算上涨概率。",
            ],
        }


def evaluate_retrieval(engine, count=100, seed=42):
    """Same held-out queries/candidate set for every method; paired weekly blocks."""
    bounds = split_boundaries(engine.series)
    if engine.meta["model_available_after"]:
        bounds["val_until"] = engine.meta["model_available_after"]
    pool = [
        j
        for j, (i, r) in enumerate(zip(engine.sid, engine.rows, strict=True))
        if engine.series[i].sessions[r] > np.datetime64(bounds["val_until"])
    ]
    rng = np.random.default_rng(seed)
    chosen = rng.choice(pool, min(count, len(pool)), replace=False) if pool else []
    results, details = {m: [] for m in engine.methods}, []
    for j in chosen:
        i, row = int(engine.sid[j]), int(engine.rows[j])
        s = engine.series[i]
        record = {"key": s.key, "row": row, "session": str(s.sessions[row]), "methods": {}}
        for method in engine.methods:
            hits = engine.retrieve(i, row, method, 8)
            if not hits:
                continue
            similarity = [
                float(
                    np.mean(
                        engine.series[engine.sid[h]].labels["direction"][engine.rows[h]]
                        == s.labels["direction"][row]
                    )
                )
                for h, _ in hits
            ]
            score = float(np.mean(similarity))
            block = str(pd.Timestamp(s.sessions[row]).to_period("W"))
            results[method].append((score, block, int(j)))
            record["methods"][method] = {"direction_agreement": score, "count": len(hits)}
        details.append(record)
    summary = {
        m: block_interval([v for v, _, _ in values], [b for _, b, _ in values], seed)
        for m, values in results.items()
    }
    comparisons = {}
    if "model" in results:
        learned = {j: (v, b) for v, b, j in results["model"]}
        for baseline in ("rule", "path"):
            pairs = [(learned[j][0] - v, b) for v, b, j in results[baseline] if j in learned]
            comparisons[f"model_minus_{baseline}"] = block_interval(
                [v for v, _ in pairs], [b for _, b in pairs], seed
            )
    return {
        "schema": SCHEMA,
        "seed": seed,
        "queries": len(chosen),
        "boundaries": bounds,
        "summary": summary,
        "paired_comparisons": comparisons,
        "details": details,
        "verdict": "No automatic graduation: rule-direction agreement is a proxy, human shape relevance and independent transfer remain required.",
    }


def blind_packet(engine, count=20, seed=42):
    """Randomized method-blind examples, with answer key kept in a separate file."""
    rng = np.random.default_rng(seed)
    bounds = split_boundaries(engine.series)
    cutoff = np.datetime64(engine.meta["model_available_after"] or bounds["val_until"])
    pool = [
        j
        for j, (i, r) in enumerate(zip(engine.sid, engine.rows, strict=True))
        if engine.series[i].sessions[r] > cutoff
    ]
    chosen = rng.choice(pool, min(count, len(pool)), replace=False) if pool else []
    packet, key = [], {}
    for number, j in enumerate(chosen):
        i, row = int(engine.sid[j]), int(engine.rows[j])
        alternatives = []
        for method in engine.methods:
            hits = engine.retrieve(i, row, method, 1)
            if hits:
                h, _ = hits[0]
                snap = engine.snapshot(int(engine.sid[h]), int(engine.rows[h]))
                alternatives.append((method, snap["bars"]))
        rng.shuffle(alternatives)
        query = engine.snapshot(i, row)
        case_id = f"B{number + 1:03d}"
        packet.append(
            {
                "id": case_id,
                "query": query["bars"],
                "alternatives": [
                    {"id": chr(65 + k), "bars": bars} for k, (_, bars) in enumerate(alternatives)
                ],
                "criteria": "只看结构相似，不看未来结果；分别填0不相关/1部分相关/2高度相关；允许都不相关。",
            }
        )
        key[case_id] = {
            "methods": {chr(65 + k): method for k, (method, _) in enumerate(alternatives)},
            "block": str(pd.Timestamp(engine.series[i].sessions[row]).to_period("W")),
        }
    return packet, key


def score_blind(ratings_path, answer_key_path):
    """Score independent human relevance; leave missing ratings missing, never zero."""
    ratings = pd.read_csv(ratings_path)
    key = json.loads(Path(answer_key_path).read_text())
    if ratings.duplicated(["case_id", "alternative"]).any():
        raise ValueError("Duplicate case/alternative ratings")
    methods = {}
    for row in ratings.to_dict("records"):
        case, alt, value = row["case_id"], row["alternative"], row["relevance_0_1_2"]
        if case not in key or alt not in key[case]["methods"]:
            raise ValueError(f"Unknown blind case/alternative: {case}/{alt}")
        if pd.isna(value):
            continue
        if value not in (0, 1, 2):
            raise ValueError("Relevance must be 0, 1 or 2")
        method = key[case]["methods"][alt]
        methods.setdefault(method, {})[case] = (float(value), key[case]["block"])
    summary = {
        m: block_interval([v for v, _ in rows.values()], [b for _, b in rows.values()])
        for m, rows in methods.items()
    }
    paired = {}
    for baseline in ("rule", "path"):
        learned = methods.get("model", {})
        pairs = [
            (learned[c][0] - v, b)
            for c, (v, b) in methods.get(baseline, {}).items()
            if c in learned
        ]
        paired[f"model_minus_{baseline}"] = block_interval(
            [v for v, _ in pairs], [b for _, b in pairs]
        )
    return {
        "summary": summary,
        "paired_comparisons": paired,
        "ratings": sum(len(v) for v in methods.values()),
        "interpretation": "Human structure relevance, independent of future returns. Missing ratings excluded; intervals use query-week blocks.",
    }
