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

## Reference baseline (local engine)

Date: 2026-07-28

Command:

```bash
./bench run 11 --solution reference --repeat 6 --engine local \
  --timeout 300 --output results/task-11-reference-baseline-local
```

Reference SHA-256:
`087c7a2894b4f0383bfc476f835933940cdfd2d9812f814adede3a39375b3f00`

Evaluator SHA-256:
`de70880ec00a86a7123aed14651b33401a7f872f667fb1598bd3ba191e29353b`

Host fingerprint:
`748423c1790b38ddbdd8eb77499b222a173b313f350e3bc35402ee8889a49dc4`
(4 vCPU Intel Xeon x86_64, 15 GiB RAM, Linux 6.12, Python 3.12.3, JAX 0.10.0,
`tensorcircuit-nightly==1.7.0.dev20260618`)

Immutable report: `results/task-11-reference-baseline-local/results.json`
(untracked; retained on the campaign host)

Report SHA-256:
`b4ce3d311a0d4ba74ec094b8330824d07135a82f5a23c44aece9a3e2d25f7d91`

Summary SHA-256:
`62f85a13ce89a20c50f26b19e294827fd8736100e1aed4003937a5702903f7e5`

```text
terminal_status: SUCCESS x 6
valid: 6/6 (Overall: PASS in every cell)
timed_out: 0
runtime_sec: 166.834263, 167.478581, 167.850734, 166.478048, 167.475835, 170.601593
mean_runtime_sec: 167.786509
median_runtime_sec: 167.477208
sample_stdev_sec: 1.465538
stderr_sec: 0.598303
min_sec: 166.478048
max_sec: 170.601593
```

Decision: `baseline`

Context only: the shared-container Docker bootstrap of 2026-07-27 measured
the same immutable reference at 153.612 ± 5.039 s under an 8-CPU/9-GiB
container on the maintainer host (`baselines/bootstrap-2026-07-27.md`), and
the ORBIT-Q publication record lists 68.10 s on an unspecified host
(`baselines/historical.json`). These numbers are not pooled with this
campaign's measurements.

## Experiment `e01`

Branch: `cursor/task-11-fused-layer-campaign-f598`

Worktree: single working clone on the campaign host (deviation recorded under
"Campaign selection and provenance").

## Hypothesis

Exact gate fusion (compose per-site rz/ry/rz into one 3x3 unitary and absorb
each even-bond pair into its entangler, dropping dense-state passes from 47
to 11 per layer), plus one batched fixed 2^5 scaling-and-squaring diagonal
Pade(3,3) exponential for every 9x9 entangler of a layer, plus a precomputed
per-basis-state coefficient vector for the diagonal single-ion term, plus a
whole-training `jax.lax.scan` with a jitted post-training readout, reduces
the evaluator-reported runtime by a statistically significant factor while
preserving the algebraic layer unitaries and reproducing the reference
energy trajectory within complex64 round-off amplified by optimizer dynamics
(falsified if any functional gate fails, if the paired-speedup CI includes
1.0, or if early-step energy deltas exceed the round-off envelope audited in
`profiles/equivalence-check.json`).

## Parent commit and diff digest

Latest accepted parent commit:
`c29a139` (`data: release task 11 public workload v3`; campaign base
`ed382bf042ecb1c87b399acaadec6bce74368649`)

Hypothesis commit: `3dbe4a4c662335600fd014126d775344445d4aad`

Candidate file: `src/solutions/task-11/solution_11.py`

Candidate SHA-256:
`d5faedc2859705be1b5a259f11ae78886dfeee9f5563eade841bedd88359e5dc`

Diff SHA-256:
`3c6da28ef2f60be4abd390e1c5d9b0244e649fa23ff0d15a0c9dce0262dbef5c`

## Data used

Public dataset version: `orbitq-workloads-v20260728.3`

Public manifest SHA-256:
`97bf9fba6626a77d8bfe734b65c30cd319587185d8cb63b0b51ebfc9e176164a`

Private evaluation used: `no`

## Command, seed, and environment

Benchmark command:

```bash
./bench run 11 --solution optimized --compare-to reference --repeat 6 \
  --engine local --timeout 300 \
  --output results/task-11-e01-fused-pairs-local
```

Public seed or case selector: canonical fixed configuration, seed 2041.

Reference SHA-256:
`087c7a2894b4f0383bfc476f835933940cdfd2d9812f814adede3a39375b3f00`

Evaluator SHA-256:
`de70880ec00a86a7123aed14651b33401a7f872f667fb1598bd3ba191e29353b`

Docker image ID: none (`--engine local`; no Docker daemon on this host)

Container session ID: none (fresh local evaluator process per cell)

Pair-order pattern: odd pairs `reference -> candidate`; even pairs
`candidate -> reference`

TensorCircuit-NG commit/version: `tensorcircuit-nightly==1.7.0.dev20260618`
(+ `envs/tensorcircuit-py311/sitecustomize.py` OMECo shortcut backport)

JAX/JAXLIB versions: `0.10.0` / `0.10.0`

## Hardware and five-minute cap

Host fingerprint:
`748423c1790b38ddbdd8eb77499b222a173b313f350e3bc35402ee8889a49dc4`

CPU allocation: 4 vCPU Intel Xeon x86_64 (whole host; no container pinning)

Memory allocation: 15 GiB host memory (no container limit)

Timeout: `300 seconds`

Measured region: evaluator-reported `End-to-end solution time`
(`run_solution(config)` only)

## Result: validity, runtime, and improvement

Immutable report: `results/task-11-e01-fused-pairs-local/results.json`
(untracked; retained on the campaign host)

Report SHA-256:
`65bc5909d6939deda5f0346ccfefa19cb07afd6b6a866960ac6b746a2e64440e`

Summary SHA-256:
`80b810cdf62fa247431104c8f44a30ca9494adca9d3d17c4d59eba432f0fbf67`

```text
terminal_status: SUCCESS x 12 (6 reference cells, 6 candidate cells)
valid: 12/12 (Overall: PASS in every cell)
timed_out: 0
passing_pairs: 6/6 (candidate wins 6/6)
reference_mean_runtime_sec: 168.361539
reference_runtime_stderr_sec: 0.367342
reference_median_runtime_sec: 168.568934
candidate_mean_runtime_sec: 114.968325
candidate_runtime_stderr_sec: 0.252797
candidate_median_runtime_sec: 115.133288
improvement_pct: 31.713427
improvement_pct_stderr: 0.129555
speedup: 1.464430
speedup_stderr: 0.002767
paired_speedup_ci_low: 1.457317
paired_speedup_ci_high: 1.471543
```

Pairwise runtimes (reference, candidate, speedup): (169.633, 115.438, 1.4695),
(168.531, 115.399, 1.4604), (168.606, 114.940, 1.4669), (167.352, 115.211, 1.4526),
(168.763, 115.056, 1.4668), (167.284, 113.767, 1.4704).

Note: the paired benchmark was executed against the exact candidate bytes
recorded in the hypothesis commit; the immutable report's per-row
`source_sha256` values equal the candidate and reference hashes above.

Trajectory equivalence (`profiles/equivalence-check.json`): max absolute
energy-density delta over the first 5/20/50/100 steps is
2.8e-5 / 6.5e-5 / 8.0e-5 / 8.0e-5 (complex64 noise floor amplified by
optimizer dynamics; exact gate fusion changes float rounding order, so the
first steps are not bit-identical). The maximum entangler generator-norm
bound over all 500 steps is 2.959 (scaled norm 0.0925 after 2^-5), inside
the fixed-Pade accuracy envelope.

Decision: `keep` (all validity gates pass; six eligible counterbalanced
local-engine pairs; candidate mean and median lower; 6/6 pairs won; 95%
paired-speedup CI excludes 1.0. The `GOAL.md` Docker promotion gate stays
closed on this host and is deferred to a Docker-capable rerun.)

## Failure signal and interpretation

No failures, timeouts, or invalid cells in this session.

## Next pivot

No further candidate iterations planned for this campaign; request a
Docker-engine rerun for formal promotion. Residual headroom is bounded by
the bandwidth-bound dense-state floor measured in
`profiles/gate-application-microbench.json` (~1.7x below the framework
circuit path for a bare reshape-matmul probe that is not shippable under
the framework-fidelity rules).

## Append-only corrections

Date: 2026-07-28

Style-only revision of `src/solutions/task-11/solution_11.py` (no protocol
change): restore the immutable reference module docstring verbatim, and replace
direct `jax` / `jax.numpy` imports with TensorCircuit backend primitives
(`K.einsum`, `K.stack`, `K.solve`, `K.jit`, `K.jaxy_scan`, etc.). Candidate
SHA-256 updated to
`d5faedc2859705be1b5a259f11ae78886dfeee9f5563eade841bedd88359e5dc`.
Smoke re-check under the same pinned local engine: 1/1 PASS at 114.687 s
(inside the prior paired-session band). The six-pair statistics recorded
above remain the campaign evidence; this revision does not reopen a new
paired session.

Date: 2026-07-28 (minimal-diff rewrite)

Rewrite `src/solutions/task-11/solution_11.py` onto the immutable reference
skeleton, keeping only the four performance changes: (1) exact gate fusion in
`apply_layer`, (2) batched fixed-order Pade entanglers, (3) diagonal onsite
coefficient vector, (4) `K.jaxy_scan` + `K.jit` training/finalize. Helpers,
naming (`spin1_even`/`spin1_odd`), initialization, and module docstring match
the reference. Candidate SHA-256 `d5faedc2859705be1b5a259f11ae78886dfeee9f5563eade841bedd88359e5dc`. Static policy line_count=198.
Smoke: 1/1 PASS at 113.573 s.

## Post-merge component ablation

Date: 2026-07-29

`profile_factor_ablation.py` measured three independently removable factors
inside the merged candidate on the 6-CPU/7-GiB Docker backend:

- fixed batched Pade entanglers execute `1.695x` faster than the batched
  adaptive-`K.expm` alternative at `4.68e-6` maximum gate error;
- the exact diagonal onsite vector executes `81.21x` faster than twelve
  framework expectations for that isolated term, with zero value error;
- ten identical candidate updates take `1.839 s` in one scan versus
  `2.130 s` as repeated compiled-step dispatches; histories are bitwise equal.

These component ratios are not multiplied or relabeled as canonical
end-to-end speedups. The report now distinguishes dominant dense-state/onsite
bandwidth savings from the smaller scan contribution.
