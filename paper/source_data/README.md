# Source-data notes

This directory contains the compact source tables for the paper update.

- `benchmark_models.csv`: Sol high/ultra, Terra high, Luna high, and DeepSeek
  V4 Flash high only.
- `task_outcomes.csv`: 5 configurations × 12 tasks. `raw_reward` preserves the
  verifier result; `final_pass` records the final human-expert adjudication.
- `expert_optimization.csv`: public expert and optimized runtimes plus the
  dominant factor reported by each upstream Task PR.
- `insights.csv`: short take-home messages for the expert optimizations.

Task 08 is final `F` for all five solver configurations. Luna and Sol ultra
retain raw reward `1` in `task_outcomes.csv`, while `final_pass=0` records the
paper-facing review decision.

Task 05 uses the legal TensorCircuit-native result from
[ORBIT-Q PR #19](https://github.com/sxzgroup/ORBIT-Q/pull/19): mean paired
end-to-end speedup `1.939×` (median `1.668×`), with OMECo `4×4` path search as
the dominant resolved factor (`1.420×`). The earlier custom no-QR MPS result is
excluded from this paper summary.

Runtime fields are seconds in `expert_optimization.csv` and minutes where
`benchmark_models.csv` uses a `_min` suffix. Missing same-machine artifact
ratios remain blank rather than being inferred from different campaigns.
