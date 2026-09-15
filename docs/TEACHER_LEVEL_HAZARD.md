# Teacher-level Hazard v1

This branch adds an opt-in competing-risk objective. The production/E3 path is
unchanged unless `--hazard-task` is explicitly supplied.

## Model

The existing causal KLineTransformer still encodes the observed history. A
four-bin head predicts conditional hazards with class order:

```text
0 = survive this bin without a first touch
1 = first upper touch in this bin
2 = first lower touch in this bin
```

The hazards are aggregated into the public Obson class order `[down, none, up]`:

```text
P(up)   = sum survival_before_bin * hazard_up
P(down) = sum survival_before_bin * hazard_down
P(none) = survival_after_last_bin
```

This avoids trying to recover first-touch order from independent upper/lower
excursion marginals. The existing `class_head` remains constructed for state
dict compatibility, but hazard mode uses the aggregated hazard probabilities as
the public logits.

## Labels and loss

`KLineDataset` derives four future-bin labels from the existing first-passage
scan. A no-touch sample is censored with four survival labels. A first-touch
sample has survival labels through the event bin and one upper/lower event label
in that bin; later bins are inactive. A same-bar double touch is ambiguous and
is masked for the hazard loss.

The loss is discrete competing-risk negative log likelihood. First-touch bins
use an event weight (default `5.0`) so that the abundant censored/survival
examples cannot make an all-survival predictor look good. No future value is
fed as an input feature, and contract segment boundaries remain enforced by the
existing dataset path.

## Run a controlled experiment

Keep the contract data, theta, symbols, periods, optimizer, and production
harness fixed. Use only:

```bash
PYTHONPATH=src python -u scripts/train_multi_symbol.py \
  --task classify --label-anchor day_close --theta-mode dynamic --theta-q 0.90 \
  --contract-mode --hazard-task --periods 60 30 \
  --daily-bars 20 --foreign-bars 20 --batch-size 256 --lr 1e-3 \
  --hazard-event-weight 5.0 \
  --epochs 40 --patience 8 --seed 42 \
  --save-dir checkpoints/hazard_s42
```

Run seed 7 only after the seed 42 smoke test builds and trains. Do not combine
`--hazard-task` with E3/path/utility heads in the first comparison; otherwise
the task change is no longer isolated.

## Decision discipline

First inspect hazard NLL, event-bin calibration, and whether the aggregated
probabilities are non-degenerate. Then run the same v2 single-position,
double-cost, and forward-confirmation harness as E3. Hazard mode is an
experiment, not a production default.
