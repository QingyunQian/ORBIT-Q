# Challenge 04 optimized expert solution

This directory packages the final reviewed Task 04 campaign from Benchmark
PR [#15](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/15),
with the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_4_fused_kraus.py` is the optimized TensorCircuit-NG variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/results-20260729/`, `research/profiles/`, and
  `research/figures/` contain the sanitized measurements and ablations.

All six matched pairs passed. Mean runtime changed from `14.742286 s` to
`5.672258 s`; mean paired speedup was `2.602133x` with a 95% t-interval of
`[2.529463x, 2.674804x]`.

The canonical expert under `tasks/challenge-04/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
