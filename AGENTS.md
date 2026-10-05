# Babel research work

These rules apply to Babel research, its code, reports and experiment plans. They
record the user's explicit workflow requirements. Other project work is outside
this research protocol.

## Read before acting

- `docs/BABEL_ROADMAP.md` and `docs/BABEL_ROADMAP.json`: current requirements,
  fixed task IDs, status, dependencies and retained future hypotheses.
- `docs/BABEL_EXPERIMENT_WORKFLOW.md`: coding, report review and completion rules.
- The protocol and evidence linked by the active task. Historical log entries do
  not override the current roadmap or the user's newer instructions.

## Every code or experiment-plan change

1. Tell the user the task ID(s), concrete question and intended deliverable before
   editing. Necessary controls may address more than one ID in the same experiment.
2. State source model, data/context, intended variable, matched controls, budget,
   selection rules, acceptance and stopping criteria before launching training.
3. After implementation or planning, record a review covering final purpose,
   model architecture, loss/training, validation, data and engineering. Record
   evidence and remaining gaps, not a blanket claim of correctness.
4. Update the roadmap and append a review event to
   `docs/BABEL_PROGRESS_LOG.jsonl`. Keep code ready, remote execution complete,
   and scientific acceptance distinct. Do not mark real-model acceptance from
   synthetic tests or arithmetic checks.
5. Preserve previous failures, task IDs, retained hypotheses and user changes.
   Changing order, budget, thresholds or source needs an explicit recorded reason;
   do not silently promote candidates or turn optional branches into active jobs.

## Every user-provided report archive

Follow the workflow: identify the archive by SHA256, source/run and protocol;
review completion, evidence and limits; map results to task IDs; update conclusions,
decision impact and next actions in the roadmap and append the review event.
Duplicate uploads are not new experiments. Failed/incomplete reports must also be
recorded, without inventing scientific results or relaxing gates to pass.

## Execution conventions

- Real neural training and real-weight execution belong on AutoDL CUDA; local
  synthetic tests and CPU report/data arithmetic are allowed.
- Work on `features/babel` unless the user changes this. Authorized preparation
  includes committing/pushing completed work; preserve unrelated user changes.
- Provide complete run/export commands only when entrypoints are ready. Export
  success, failed and stopped reports to `/root/autodl-tmp/download`.
- Prefer matched controls in one bounded release. Respect existing source hashes;
  new modules are preferable to silently changing historical bound code.
- Do not require repeated user approval for already authorized work. Do not spawn
  agents unless the user or other applicable instructions explicitly request it.

## Documentation consolidation (user clarification, 2026-10-05)

- Use `docs/BABEL_ROADMAP.md` as the single execution and conclusion-disposition entry;
  its JSON is a synchronized mirror. Update existing evidence documents when needed.
- Do not create another explanatory/clarification/plan document for status corrections.
  Preserve historical bound protocols and report evidence; do not rewrite their hashes.
- Current priority is correcting original errors, verifying affected conclusions, then
  archiving reliable conclusions. Model improvement is not the closure criterion for
  correcting an unsupported historical claim. No new optimization branch before that work.
