# Task 11 Autoresearch Campaign

Destination: `research/task-11/LOG.md`

Task: `task-11`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and provenance

Selected task: `task-11` (spin-1 Haldane-chain VQE with string-order
readout).

Live open pull requests inspected on 2026-07-28: on `sxzgroup/ORBIT-Q`,
`#2` (ForgeCode agent solver), `#3` (scoring-policy fix), `#4` (Fable 5
agent-axis run record), `#5` (GPT-5.6 Sol benchmark results); on this
repository, `#4` (Task 01 MPO energy, a task-01 campaign). None is an
active Task 11 solution-improvement PR, so Task 11 is eligible for this
campaign. The Task 12 campaign of this repository closed with merge commit
`ed382bf042ecb1c87b399acaadec6bce74368649`, which is this campaign's base.

Precursor research disclosure: the bottleneck profiling and candidate design
were first executed on 2026-07-28 against the byte-equivalent ORBIT-Q
publication reference in the fork `QingyunQian/ORBIT-Q` (branch
`cursor/optimize-challenge-11-f598`, `optimized_sloutions/challenge-11/`,
PR #5 there), on the same 4 vCPU cloud VM with
`tensorcircuit-nightly==1.8.0.dev20260726` (external context: reference
168.574 ± 1.298 s vs candidate 118.200 ± 3.735 s over five interleaved
official-evaluator trials, all PASS). All figures used for claims in this
repository were re-measured here in the pinned lock environment; external
numbers are context only and are never pooled with in-repo measurements.

Campaign workspace deviation: single working clone on the campaign host with
branch `cursor/task-11-fused-layer-campaign-f598` (Cursor cloud-agent
branch-naming policy) instead of the `codex/orbitbreakers/task-11/<id>`
worktree layout. Exactly one hypothesis edits
`src/solutions/task-11/solution_11.py`.

Engine deviation: the campaign host has no Docker daemon, so every in-repo
measurement uses `./bench ... --engine local` with a virtual environment
installed exactly from `envs/tensorcircuit-py311/requirements.lock`
(Python 3.12.3), the environment `sitecustomize.py` on `PYTHONPATH`, and
`NUMBA_DISABLE_JIT=1`. The `GOAL.md` Gate 3 Docker protocol therefore stays
closed on this host; local-engine paired evidence is recorded below and a
Docker rerun is requested in `IMPLEMENTATION_COMPARISON.md`.

## Reference bottleneck profile

Date: 2026-07-28

Scripts: `research/task-11/profile_reference.py`,
`research/task-11/profile_gate_application.py` (pinned lock environment).
Sanitized outputs: `profiles/reference-profile.json`,
`profiles/gate-application-microbench.json`.

Findings recorded in `SURVEY.md`: the reference spends 1.32 s on jit trace
and 2.73 s on XLA compile (8135-line StableHLO), then runs the 500-step
loop at 323 ms per step (~97% of end-to-end time). Forward `build_state`
costs 81 ms (5 layers x 47 dense-state gate applications), forward energy
36 ms (23 separate `expectation` contractions). One 9x9 two-qudit
contraction against the dense 4.25 MB state costs 0.5-2.9 ms depending on
bond position (transpose+gemm lowering of strided-axis einsums); a bare
reshape-matmul floor probe shows only ~1.7x headroom below the framework
circuit path and is a boundary datapoint only. The workload is
memory-bandwidth-bound: the achievable gain comes from cutting dense-state
passes, not from compilation or dispatch.
