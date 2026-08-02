# Source-data notes

- `paper_agent_axis.csv`: original paper agent-axis totals plus Sol high/ultra,
  Terra high, Luna high, and DeepSeek V4 Flash high. It drives updated Fig. 1c,
  Fig. 2b, and the top row of Fig. 4.
- `paper_framework_axis.csv`: original paper framework-axis totals for the
  unchanged bottom row of Fig. 4.
- `paper_expert_runtimes.csv`: original public per-task TensorCircuit expert
  references used to recompute the new Fig. 2b geometric means.
- `benchmark_models.csv`: compact new-campaign aggregate table.
- `task_outcomes.csv`: 5 configurations × 12 tasks. `raw_reward` preserves the
  verifier result; `final_pass` records final human-expert adjudication.
- `expert_optimization.csv`: original and optimized expert runtimes plus the
  dominant factor retained after ablation.
- `insights.csv`: short take-home messages for the expert optimizations.

The original values come from the main and supplementary tables of
[arXiv:2607.03105](https://arxiv.org/abs/2607.03105). The new campaign totals
come from ORBIT-Q PRs #5, #6, #20, #21, and #23. Fig. 2b's new runtime ratios
are recomputed against the original paper's public per-task expert
TensorCircuit runtimes, over final passed tasks only.

Task 08 is final `F` for all five new solver configurations. Luna and Sol ultra
retain raw reward `1` in `task_outcomes.csv`; `final_pass=0` records the
paper-facing decision. Task 05 uses the legal TensorCircuit-native result from
[PR #19](https://github.com/sxzgroup/ORBIT-Q/pull/19): **1.939×** mean paired
end-to-end speedup (median **1.668×**). The earlier custom no-QR MPS result is
excluded.
