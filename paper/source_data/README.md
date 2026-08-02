# Source-data provenance

These CSV files are the compact paper-facing extraction of the benchmark and
expert-optimization artifacts. They are intentionally human-readable and
contain no hidden baseline values.

## Files

- `benchmark_models.csv`: one row per solver configuration. `passes` and
  `failures` are final paper counts. Wall time, tokens, and cost come from the
  solving-side summaries. `gm_slowdown` is populated only when a committed,
  same-machine expert-reference comparison exists.
- `task_outcomes.csv`: one row per model/task. `raw_reward` is the verifier
  reward; `final_pass` is the final paper adjudication. Functional/static/audit
  fields are retained so a raw reward can be explained rather than silently
  overwritten.
- `expert_optimization.csv`: public human-expert baseline versus optimized
  TensorCircuit-NG runtime, with the full end-to-end ratio and the dominant
  factor from the corresponding ablation.
- `insights.csv`: short mechanistic take-home messages used in Figure 5.

## Final-review overrides

The benchmark's raw verifier artifacts are preserved. The final paper matrix
applies the later human-expert review requested for the comparison:

- Task 08 is final `F` for GPT-5.6 Sol high, Sol ultra, Terra high, Luna high,
  DeepSeek V4 Flash high, and DeepSeek V4 Flash max.
- Luna Task 08 retains raw reward `1`, functional/static/audit `1`, and its
  measured runtime in `task_outcomes.csv`; only `final_pass` is changed to `0`.
- Sol ultra Task 08 is treated analogously; its source reward/audit artifact is
  retained while the final reviewed result is `F`.
- Sol ultra Task 07 uses the source-adjudicated pass despite a raw audit
  discrepancy, as recorded by the upstream PR history.
- Fable 5 remains the reported 12/12 artifact result and is not placed on the
  comparable Docker-agent resource axis.

These are review labels, not regenerated verifier runs. Any manuscript claim
that needs the original verifier-only view should use `raw_reward` and the
functional/static/audit columns explicitly.

## Units and comparison boundaries

- Runtime columns are seconds in `expert_optimization.csv` and minutes in
  `benchmark_models.csv` where the header says `_min`.
- Token columns are millions of solving-side tokens; cost is recorded USD from
  the corresponding run summaries.
- Missing same-machine ratios are blank by design. They must not be replaced
  by ratios inferred from different hosts, models, or harnesses.
- Fable's Cursor/Fable resources are not comparable to Harbor Docker-agent
  resources.

The public design context is the [ORBIT-Q repository](https://github.com/sxzgroup/ORBIT-Q)
and [arXiv:2607.03105](https://arxiv.org/abs/2607.03105).
