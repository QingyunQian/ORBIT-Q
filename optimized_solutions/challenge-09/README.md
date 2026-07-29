# Challenge 09 optimized expert solution

This directory packages the final reviewed Task 09 campaign from Benchmark
PRs [#2](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/2),
[#5](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/5), and
[#9](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/9), with
the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_9_causal_cone.py` is the optimized TensorCircuit-NG variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the complete comparison and
  semantic audit, including the final six-pair addendum.
- `research/profiles/`, `research/figures/`, and the plotting script preserve
  the post-PR factor screens.

All six final matched pairs passed. Mean runtime changed from `33.503727 s`
to `8.766516 s`; mean paired speedup was `3.821725x` with a 95% t-interval
of `[3.729633x, 3.913817x]`.

The canonical expert under `tasks/challenge-09/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
