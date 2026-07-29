# Challenge 05 optimized expert solution

This directory packages the final reviewed Task 05 campaign from Benchmark
PRs [#3](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/3) and
[#14](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/14), with
the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_5_exact_mps.py` is the final exact no-QR MPS
  TensorCircuit-NG variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/results-20260729/`, `research/profiles/`, and the accompanying
  scripts contain equivalence checks, component timings, and ablation plots.

All six matched pairs passed. Mean runtime changed from `97.083712 s` to
`6.898409 s`; mean paired speedup was `14.075570x` with a 95% t-interval of
`[13.675752x, 14.475387x]`.

The canonical expert under `tasks/challenge-05/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
