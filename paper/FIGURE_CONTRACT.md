# Figure contract

The paper update reuses existing ORBIT-Q figure assets instead of introducing
a new visual language.

1. The original paper figures remain unchanged in `docs/assets/`.
2. The five-model task-outcome and agent-resource figures are the exact assets
   from upstream PR #23. They contain Sol high/ultra, Terra high, Luna high,
   and DeepSeek V4 Flash high only.
3. Human-expert optimization figures are the exact `factor-ablation.svg`
   assets from the corresponding Task PRs. Task 05 uses the legal
   TensorCircuit-native PR #19 figure.

No Fable 5 or DeepSeek max result is included. No missing same-machine runtime
ratio is inferred. Agent wall time, solver token use, artifact runtime, and
expert-optimization speedup remain distinct quantities.
