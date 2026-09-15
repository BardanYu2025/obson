# KLM: K-Line Large Model experiment protocol

## Purpose

KLM tests whether a market-specific encoder plus future queries can learn
useful continuous path information and improve the existing three-class
trading task. It is an experimental branch; the E3 production model remains
unchanged unless a full gate is passed.

## Model

With `--klm-task`, the history encoder is bidirectional because its input is
the fully observed, already-closed history window. Four learned future
queries cross-attend the encoder memory. Each query predicts three targets at
three quantiles (`q10`, `q50`, `q90`):

1. close return relative to the sample's current price;
2. maximum favorable upward excursion;
3. maximum adverse downward excursion.

All targets are divided by the sample theta. The horizons are `[1, 2, 4,
close]` in valid future bars. A horizon that does not exist is masked. The
close horizon is the actual label anchor, not an artificial fourth fraction.

The trade query uses the pooled history representation plus the mean future
query representation and predicts `[down, none, up]`. Real future targets
are used only in the loss and never as features.

## Loss

The production classification objective remains the champion objective:

```text
L_trade = 0.5 * weighted hard CE + 0.5 * soft-label CE
L_klm   = masked pinball loss for q10/q50/q90
L_total = L_trade + klm_reg_loss_weight * L_klm
```

The quantile head is parameterized to guarantee `q10 <= q50 <= q90`.

## Gates

### Gate 0: smoke

Stop immediately on NaN, all-invalid horizon masks, non-finite loss, or
constant trade output. Report the valid-mask rate per horizon and all three
losses.

### Gate 1: seed42 feasibility

Run the full time split. The candidate must satisfy all of:

- finite quantile loss and sensible interval coverage on validation;
- validation `mean_edge` is not materially below the random-initialized
  classifier baseline;
- no systematic disappearance of either direction;
- production test report is non-negative after the documented cost model;
- no single product or short interval explains the whole result.

If any item fails, do not run seed7. Record the branch as a failed probe.

### Gate 2: seed7 stability

Only after Gate 1, run seed7 with identical data and parameters. Require the
two seeds to be directionally consistent, then run the frozen production
single-position backtest, doubled-cost check, drawdown, and key-product
breakdown. KLM is not a champion unless it improves the production result
without materially worsening stability or drawdown.

Quantile metrics are diagnostic and do not replace production selection.
