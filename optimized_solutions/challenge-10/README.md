# Challenge 10 optimized expert solution

This directory packages the final reviewed Task 10 campaign from Benchmark
PR [#8](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/8),
with the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_10_exact_mps.py` is the exact bounded-rank MPS/MPO
  TensorCircuit-NG variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/profiles/`, `research/figures/`, and the factor-ablation runner
  preserve the profiling, five-pair removal ablations, and six-pair result.

The requested five-pair view passed in all ten cells. Mean runtime changed
from `18.931296 s` to `3.869287 s`; mean paired speedup was `4.898251x`
with a 95% t-interval of `[4.597784x, 5.198719x]`. The supplemental sixth
pair also passed.

The canonical expert under `tasks/challenge-10/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
