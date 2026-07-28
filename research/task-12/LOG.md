# Task 12 Autoresearch Campaign

Destination: `research/task-12/LOG.md`

Task: `task-12`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and provenance

Selected task: `task-12` (variational circuit to MPS overlap optimization).

Live open pull requests on `sxzgroup/ORBIT-Q` inspected on 2026-07-28:
`#2` (ForgeCode agent solver), `#3` (scoring-policy fix), `#4` (Fable 5
agent-axis run record), `#5` (GPT-5.6 Sol benchmark results). None is an
active Task 12 solution-improvement PR, so Task 12 is eligible for this
campaign.

Precursor research disclosure: the bottleneck profiling and candidate design
for this campaign were first executed on 2026-07-28 against the
byte-equivalent ORBIT-Q publication reference in the fork
`QingyunQian/ORBIT-Q` (branch `cursor/optimize-challenge-12-f598`,
`optimized_sloutions/challenge-12/`), on the same 4 vCPU cloud VM but with
`tensorcircuit-nightly==1.8.0.dev20260726`. All figures used for claims in
this repository were re-measured here in the pinned lock environment; the
external numbers are context only and are never pooled with in-repo
measurements. See `SURVEY.md` for the porting statement.

Campaign workspace deviation: this campaign runs in a single working clone on
the campaign host with branch `cursor/task-12-batched-su4-campaign-f598`
(Cursor cloud-agent branch-naming policy) instead of the
`codex/orbitbreakers/task-12/<opaque-id>` worktree-per-hypothesis layout.
Exactly one hypothesis edits `src/solutions/task-12/solution_12.py`, so the
one-worktree-per-hypothesis isolation is preserved in substance.

Engine deviation: the campaign host has no Docker daemon, so every in-repo
measurement uses `./bench ... --engine local` with a virtual environment
installed exactly from `envs/tensorcircuit-py311/requirements.lock`
(Python 3.12.3), the environment `sitecustomize.py` on `PYTHONPATH`, and
`NUMBA_DISABLE_JIT=1`. The `GOAL.md` Gate 3 Docker protocol therefore stays
closed on this host; local-engine paired evidence is recorded below and a
Docker rerun is requested in `IMPLEMENTATION_COMPARISON.md`.

## Reference baseline (local engine)

Date: 2026-07-28

Command:

```bash
./bench run 12 --solution reference --repeat 6 --engine local \
  --timeout 300 --output results/task-12-reference-baseline-local
```

Reference SHA-256:
`10cfd516bc250633f4675653e0d8986002e56f4d5916a9c2972c1085193f5d38`

Evaluator SHA-256:
`08940a5fabfd88a957c467edabfbe6faa7b766f38b4d518557e50e94fcf3b277`

Host fingerprint:
`748423c1790b38ddbdd8eb77499b222a173b313f350e3bc35402ee8889a49dc4`
(4 vCPU Intel Xeon x86_64, 15 GiB RAM, Linux 6.12, Python 3.12.3, JAX 0.10.0,
`tensorcircuit-nightly==1.7.0.dev20260618`)

Immutable report: `results/task-12-reference-baseline-local/results.json`
(untracked; retained on the campaign host)

Report SHA-256:
`6d15f64bdf03097f1423fa16e0f434c97b2d1e44dafd5973abec0e08df004975`

Summary SHA-256:
`600753c2e09b9561afdfd6f79e9c795a7a7d626d4b906bb6149b530a0c2164fe`

```text
terminal_status: SUCCESS x 6
valid: 6/6 (Overall: PASS in every cell)
timed_out: 0
runtime_sec: 9.113186, 9.188893, 9.067477, 9.064269, 9.086606, 9.181242
mean_runtime_sec: 9.116946
median_runtime_sec: 9.099896
sample_stdev_sec: 0.055618
stderr_sec: 0.022706
min_sec: 9.064269
max_sec: 9.188893
```

Decision: `baseline`

Context only: the shared-container Docker bootstrap of 2026-07-27 measured
the same immutable reference at 11.261 ± 0.972 s under an 8-CPU/9-GiB
container on the maintainer host (`baselines/bootstrap-2026-07-27.md`), and
the ORBIT-Q publication record lists 6.12 s on an unspecified host
(`baselines/historical.json`). These numbers are not pooled with this
campaign's measurements.

## Reference bottleneck profile

Date: 2026-07-28

Scripts: `research/task-12/profile_reference.py`,
`research/task-12/profile_expm.py` (pinned lock environment, local engine
conditions). Sanitized outputs:
`profiles/reference-profile.json`, `profiles/expm-microbench.json`.

Findings recorded in `SURVEY.md`: the reference spends ~0.14 s on target
conversion, ~1.7 s on jit trace, ~3.4 s on XLA compile of an 8884-line
StableHLO module, and ~3.3 s on 5000 steps at ~0.67 ms per step. A scan step
containing only the batched 31-gate build plus gradient and Adam costs
0.381 ms with the norm-adaptive `jax.scipy.linalg.expm` versus 0.123 ms with
the candidate's fixed-order diagonal Pade(3,3); the per-gate `su4`
construction chain, not the tensor-network contraction, dominates both the
graph size and the step time.

## Experiment `e01`

Branch: `cursor/task-12-batched-su4-campaign-f598`

Worktree: single working clone on the campaign host (deviation recorded under
"Campaign selection and provenance").

## Hypothesis

Building all 31 SU4 gate matrices per step with one batched einsum against
the stacked su(4) generators and one fixed 2^5 scaling-and-squaring diagonal
Pade(3,3) exponential, applying them through `circuit.any`, and running the
5000 Adam updates inside one `jax.lax.scan` reduces the evaluator-reported
runtime by at least 3x while reproducing the reference trajectory within
complex64 round-off (falsified if any functional gate fails, if the paired
speedup CI includes 1.0, or if the early-step loss trajectory deviates beyond
round-off scale).

## Parent commit and diff digest

Latest accepted parent commit:
`611c35b` (`data: release task 12 public workload v2`; campaign base
`690ffbac51715afd0a3e80718eeb6de20f11863a`)

Hypothesis commit: `ef005888101c98de30fda2df3331f5f8461bf0ab`

Candidate file: `src/solutions/task-12/solution_12.py`

Candidate SHA-256:
`1d9a36c2649e938666f680e98e423e966f82c503da64ce7e4e06c5e33344560a`

Diff SHA-256:
`0badae624ed7ec5e309e1a0684e431e407895dc54f4530d61846a3320ffd309e`

## Data used

Public dataset version: `orbitq-workloads-v20260728.2`

Public manifest SHA-256:
`569a0141c5f0d723f275112d58cc6a69a3af10ad84ab8536c072604ee838cbfd`

Private evaluation used: `no`

## Command, seed, and environment

Benchmark command:

```bash
./bench run 12 --solution optimized --compare-to reference --repeat 6 \
  --engine local --timeout 300 \
  --output results/task-12-e01-batched-pairs-local
```

Public seed or case selector: canonical fixed configuration, seed 2039.

Reference SHA-256:
`10cfd516bc250633f4675653e0d8986002e56f4d5916a9c2972c1085193f5d38`

Evaluator SHA-256:
`08940a5fabfd88a957c467edabfbe6faa7b766f38b4d518557e50e94fcf3b277`

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
(`run_solution(config)` only; evaluator-side DMRG excluded by the evaluator)

## Result: validity, runtime, and improvement

Immutable report: `results/task-12-e01-batched-pairs-local/results.json`
(untracked; retained on the campaign host)

Report SHA-256:
`653aaaaabff79a4c9a2bdcdbea4a82b561d8cff0824b83f84df08f5efb2659cd`

Summary SHA-256:
`725eeecf70b572fca9d7a4e88dbe995f216b87b75173624dedb1a75715c108f7`

```text
terminal_status: SUCCESS x 12 (6 reference cells, 6 candidate cells)
valid: 12/12 (Overall: PASS in every cell; final fidelities 0.86954-0.87016)
timed_out: 0
passing_pairs: 6/6 (candidate wins 6/6)
reference_mean_runtime_sec: 9.082742
reference_runtime_stderr_sec: 0.027423
reference_median_runtime_sec: 9.086063
candidate_mean_runtime_sec: 2.320613
candidate_runtime_stderr_sec: 0.002886
candidate_median_runtime_sec: 2.320519
improvement_pct: 74.450306
improvement_pct_stderr: 0.095146
speedup: 3.913941
speedup_stderr: 0.014572
paired_speedup_ci_low: 3.876545
paired_speedup_ci_high: 3.951460
```

Pairwise runtimes (reference, candidate, speedup): (9.132, 2.324, 3.9299),
(9.040, 2.317, 3.9023), (9.023, 2.318, 3.8926), (9.163, 2.311, 3.9651),
(9.132, 2.323, 3.9312), (9.006, 2.331, 3.8629).

Note: the paired benchmark was executed against the exact candidate bytes
recorded in the hypothesis commit; the immutable report's per-row
`source_sha256` values equal the candidate and reference hashes above.

Trajectory equivalence (`profiles/equivalence-check.json`): the first 10
recorded losses match the reference bit-for-bit in float32 (max delta 0.0);
max delta 3.98e-05 over 50 steps and 1.27e-03 over 400 steps, the ordinary
round-off scale amplified by optimizer dynamics. The maximum su(4)
generator-norm bound over all 5000 steps is 3.6016 (scaled norm 0.1125 after
2^-5), far inside the fixed-Pade accuracy envelope measured in
`profiles/expm-microbench.json`.

Decision: `keep` (all validity gates pass; six eligible counterbalanced
local-engine pairs; candidate mean and median lower; 6/6 pairs won; 95%
paired-speedup CI excludes 1.0. The `GOAL.md` Docker promotion gate stays
closed on this host and is deferred to a Docker-capable rerun.)

## Failure signal and interpretation

No failures, timeouts, or invalid cells in this session.

## Next pivot

Track the pair-fused ququart formulation as a reference-derived variant and
measure it under the same paired protocol; request a Docker-engine rerun for
formal promotion.

## Append-only corrections

None.

## Variant benchmark: pair-fused ququart contraction

Date: 2026-07-28

Variant tracking commit: `3026f94b202bf91d1d5b7c8656e8ab5f23e6a030`

Variant file: `src/solutions/task-12/variants/solution_12_fused.py`

Variant SHA-256:
`5edd437829352c573b24ae2f9021ef6046c2cf53788a8b79c8811b8fe0e3103f`

This is a tracked reference-derived variant benchmark, not a second campaign
candidate; `src/solutions/task-12/solution_12.py` remains the e01 candidate.

Command:

```bash
./bench run 12 --solution fused --compare-to reference --repeat 6 \
  --engine local --timeout 300 \
  --output results/task-12-fused-variant-pairs-local
```

Environment, host, pair-order pattern, timeout, and measured region: same as
experiment `e01`.

Immutable report: `results/task-12-fused-variant-pairs-local/results.json`
(untracked; retained on the campaign host)

Report SHA-256:
`f7bc804a975b299a786b6bc519764a2dd926076e054d82d9aea20094cc0abed8`

Summary SHA-256:
`4d6a20dd02145ef5717abcb1c9ad643f7e9dbde4a3f59bfcb7c52523fc6c6608`

```text
terminal_status: SUCCESS x 12 (6 reference cells, 6 variant cells)
valid: 12/12 (Overall: PASS in every cell; final fidelities 0.86904-0.87013)
timed_out: 0
passing_pairs: 6/6 (variant wins 6/6)
reference_mean_runtime_sec: 9.020622
reference_runtime_stderr_sec: 0.026975
reference_median_runtime_sec: 9.033664
candidate_mean_runtime_sec: 2.123708
candidate_runtime_stderr_sec: 0.004555
candidate_median_runtime_sec: 2.121520
improvement_pct: 76.457192
improvement_pct_stderr: 0.111281
speedup: 4.247582
speedup_stderr: 0.019961
paired_speedup_ci_low: 4.196459
paired_speedup_ci_high: 4.299083
```

Pairwise runtimes (reference, variant, speedup): (9.017, 2.119, 4.2557),
(9.082, 2.124, 4.2756), (9.050, 2.111, 4.2870), (8.931, 2.128, 4.1960),
(8.954, 2.143, 4.1786), (9.089, 2.117, 4.2938).

Interpretation: after e01's gate-construction fix, halving the contraction
network buys a further ~8% end to end (4.248x vs 3.914x); the contraction
was never the dominant cost. The variant stays PASS-equivalent with final
fidelities inside the reference's own run-to-run band.

Decision: `keep` as a tracked variant (same local-engine eligibility caveat
as e01).
