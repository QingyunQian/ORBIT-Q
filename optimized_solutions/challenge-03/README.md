# Challenge 03 optimized expert solution

This directory packages the final reviewed Task 03 campaign from Benchmark
PR [#16](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/16),
with the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_3_product_contraction.py` is the optimized TensorCircuit-NG
  variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/profiles/`, `research/figures/`, and the profiling/equivalence
  scripts preserve every retained factor and rejected-factor result.

All six matched pairs passed. Mean runtime changed from `4.101350 s` to
`0.924608 s`; mean paired speedup was `4.435277x` with a 95% t-interval of
`[4.287006x, 4.583549x]`.

The canonical expert under `tasks/challenge-03/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
