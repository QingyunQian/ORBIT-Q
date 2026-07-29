# Challenge 01 optimized expert solution

This directory packages the final reviewed Task 01 campaign from
[`hmyuuu/OrbitBreakersExpertBenchmarks`](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks):
PRs [#4](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/4) and
[#18](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/18), with
the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_1_graph_compression.py` is the optimized TensorCircuit-NG variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/results-20260729/`, `research/profiles/`, and
  `research/figures/` contain the sanitized measurements and factor
  ablations.
- `research/HISTORICAL_MPO_ENERGY_COMPARISON.md` retains the earlier MPO
  campaign record.

Six matched pairs passed for both implementations. Mean runtime changed from
`60.651144 s` to `6.359807 s`; mean paired speedup was `9.636410x` with a
95% t-interval of `[8.680782x, 10.592037x]`.

The canonical expert under `tasks/challenge-01/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
