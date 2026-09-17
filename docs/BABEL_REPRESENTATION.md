# Causal bar representation pilot R1

This experiment changes the objective from rule classification to causal bar
representation pretraining. It does not overwrite or load Babel v1 weights.
No claim of universal representations, market understanding or trading utility
is made. Do not select hyperparameters using the resulting test report.

## Architecture and objectives

- 4 causal Transformer layers, hidden 128, 4 heads, FFN 512, dropout 0.1.
- Every completed bar produces `model(x)["h"]`, shape batch × bars × 128.
- Context 128 or 256 bars, chosen per batch; first 64 bars warm up the context.
- Main objective: ordered 0.1/0.5/0.9 quantiles for the first seven Babel input
  features at horizons 1, 4 and 16 observed bars. Features are OHLC relative to
  their preceding close/ATR, volume z-score, OI change and OI z-score.
- Loss = forecast pinball + 0.2 current-feature SmoothL1 reconstruction
  + 0.1 mean structure-task cross entropy. Missing OI dimensions are masked.
- Reconstruction preserves selected current information; it can be easy to
  copy and does not prove temporal reasoning. No claim of preserving all history.
- Targets inherit causal feature clipping to [-20,20]. Forecasts are marginal
  distributions of normalized features, NOT a joint valid OHLC generator.
  Horizon 4/16 predicts that particular bar, not an aggregate four/sixteen-bar candle.
- Historical inputs may precede a split boundary; all supervised future targets
  remain within that split and the same contract. Main eligibility applies at
  the anchor. Existing gap and liquidity limitations remain; no automatic gap removal.

AdamW lr 3e-4, weight decay 0.01, clip gradient norm 1, seed 42, stride 16,
batch 32, 30 epochs. Validation endpoint mean pinball loss selects best.pt.
Reference Babel manifest freezes source fingerprints and date cuts. Empty output
directory required; no resume. CUDA-only runner. GPU memory use is not a quality metric.

## Evaluation

`evaluate` reads best.pt without updates. Reports test quantile loss against
constant quantiles fitted on training endpoints only, including per-feature and
per-horizon losses and supports. One fixed downstream task uses the accumulated
8-bar close change divided by current ATR, classified at -0.5/+0.5. A fixed
ridge classifier (penalty 10, training-only standardization) compares pretrained
last-bar vectors, same-architecture random last-bar vectors and raw last-bar +
last-32-mean features. This task is related to pretraining, not an independent
semantic ground truth. Raw baseline has less temporal detail than the encoder.
No hyperparameter search is performed on test data.

This is a single-seed feasibility run, not a capacity ablation. Correlated stride-16
samples are not independent observations. The previously inspected test period
is not a pristine holdout. No confidence claim or graduation threshold is supplied.
Useful evidence: lower forecast loss than training constants AND better frozen
probe performance than random/raw features; mixed results require investigation.
Neither metric alone demonstrates general usefulness. Future research needs new
time holdouts, seeds and additional unrelated downstream tasks.

## AutoDL

```bash
cd /root/autodl-tmp/obson
git pull --ff-only origin features/babel
mkdir -p logs
nohup bash -c 'bash scripts/babel_repr_autodl.sh train && bash scripts/babel_repr_autodl.sh evaluate' > logs/babel_repr_r1_s42.log 2>&1 &
tail -f logs/babel_repr_r1_s42.log
```

Default data `/root/autodl-tmp/data/contracts`; reference
`checkpoints/babel_r1_s42/manifest.json`; output
`checkpoints/babel_repr_r1_s42`. Override via `BABEL_DATA`, `BABEL_REFERENCE`,
`BABEL_REPR_RUN`, `BABEL_BATCH_SIZE`, `BABEL_EPOCHS`. Evaluation can be rerun using
`bash scripts/babel_repr_autodl.sh evaluate` without training.

Return `manifest.json`, `history.jsonl`, `representation_metrics.json` from the
new output directory. best.pt uses schema `babel-bar-representation-v1` and must
not be passed to legacy Babel index/serve commands. To extract vectors in Python,
construct `BarEncoder(Config(**checkpoint["config"]))`, load `checkpoint["model"]`,
call `.eval()` and use `torch.no_grad()` with the unchanged causal feature schema.
Only completed bars should be supplied in live inference. Each h_t depends on the
available context; chunking changes context and thus may change h_t.
