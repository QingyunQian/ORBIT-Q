# Challenge 02 optimized expert solution

This directory packages the final reviewed Task 02 campaign from Benchmark
PR [#17](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/17),
with the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_2_training_scan_batched_purity.py` is the optimized
  TensorCircuit-NG variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/profiles/`, `research/figures/`, and the plotting/equivalence
  scripts preserve the factor-by-factor ablation evidence.

All six matched pairs passed. Mean runtime changed from `4.495463 s` to
`4.031039 s`; mean paired speedup was `1.115649x` with a 95% t-interval of
`[1.057686x, 1.173613x]`.

The canonical expert under `tasks/challenge-02/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
