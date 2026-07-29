# Challenge 12 optimized expert solutions

This directory packages the final reviewed Task 12 campaign from Benchmark
PR [#6](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/6) plus
the Task 12 half of ablation PR
[#13](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/13), with
the final tree taken from Benchmark `main` at `7e2298b`.

- `solution_12_batched_su4.py` is the promoted campaign candidate.
- `solution_12_pair_fused.py` is the separately tracked, faster exact
  pair-fused variant.
- `research/IMPLEMENTATION_COMPARISON.md` is the final report.
- `research/profiles/`, `research/figures/`, and the component profiler
  preserve the post-merge factor attribution.

For upstream review, both optimized files restore the immutable expert's
module docstring verbatim. Only comments changed relative to the reviewed
Benchmark artifacts; executable code is identical.

All six local-engine matched pairs passed in both sessions. The promoted
candidate measured `3.914003x` mean paired speedup; the pair-fused variant
measured `4.247771x`. The report explicitly marks the formal Docker promotion
rerun as pending rather than overstating the claim.

The canonical expert under `tasks/challenge-12/solution/` is intentionally
unchanged. Benchmark-harness reproduction commands in the research record
should be run in the Benchmark repository pinned above.
