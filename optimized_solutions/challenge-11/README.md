# Challenge 11 optimized expert solution

This directory packages the final reviewed Task 11 campaign from Benchmark
PR [#7](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/7) plus
the Task 11 half of ablation PR
[#13](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/13), with
the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_11_fused_layers_pade.py` is the optimized TensorCircuit-NG
  variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/profiles/`, `research/figures/`, and the component profiler
  preserve the post-merge factor attribution.

All six local-engine matched pairs passed. Mean runtime changed from
`168.361539 s` to `114.968325 s`; mean paired speedup was `1.464430x` with a
95% t-interval of `[1.457317x, 1.471543x]`. The report explicitly marks the
formal Docker promotion rerun as pending rather than overstating the claim.

The canonical expert under `tasks/challenge-11/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
