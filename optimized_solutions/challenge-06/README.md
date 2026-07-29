# Challenge 06 optimized expert solution

This directory packages the final reviewed Task 06 campaign from Benchmark
PR [#12](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/12),
with the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_6_native_jaxode.py` is the optimized TensorCircuit-NG variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/profiles/`, `research/figures/`, and the validation/profiling
  scripts preserve the factor ablations and five-pair result.

All five matched pairs passed. Mean runtime changed from `41.4259 s` to
`27.5366 s`; mean paired speedup was `1.50446x` with a 95% t-interval of
`[1.48875x, 1.52018x]`.

The canonical expert under `tasks/challenge-06/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
