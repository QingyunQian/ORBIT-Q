# Task 10 Autoresearch Campaign

Task: `task-10`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and gate setup

Date: 2026-07-29

Branch: `codex/orbitbreakers/task-10/extreme-cmz`

Latest accepted parent commit:
`7be259106c9dae899e5ea8f82c2251ca285640a4`

The live open-PR search on `sxzgroup/ORBIT-Q` found whole-suite benchmark PRs
4, 5, and 6, but no active Task 10 human-expert optimization PR. This campaign
selects Task 10 and will not edit or benchmark another task.

The immutable reference and editable source were byte-identical at campaign
start:
`sha256:0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172`.

The first build attempt for the pinned
`orbitbreakers-expert-benchmarks:tensorcircuit-py311` image failed during
`apt-get update` after a network interruption. No candidate code had been
changed.

A workload-semantic validation was then run with the immutable human expert
in the already available ORBIT-Q TensorCircuit-NG image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
at 6 CPUs / 7 GiB, network disabled. The evaluator reported:

```text
End-to-end solution time: 18.117244s
Initial energy density: 0.9718865752
Final energy density: -1.1781760454
Exact ground energy density: -1.2925285569
Overall: PASS
```

This validates the canonical public configuration and output contract. It is
not a paired performance claim and is not pooled with future pinned-image
measurements.

## Experiment `e01-exact-mps-scan`

Branch: `codex/orbitbreakers/task-10/extreme-cmz`

## Hypothesis

An exact TensorCircuit-NG `MPSCircuit` formulation will retain bond dimension
at most four across the two bond-2 CMZ applications, and a whole-training scan
will eliminate host dispatch. Together they will reduce valid end-to-end
runtime relative to the immutable general-network/OMECo expert.

## Parent commit and diff digest

Latest accepted parent commit:
`7be259106c9dae899e5ea8f82c2251ca285640a4`

Hypothesis commit: pending gate completion

Candidate file: `src/solutions/task-10/solution_10.py`

Candidate SHA-256: pending

Diff SHA-256: pending

## Data used

Public dataset version: pending Task 10 gate artifact

Private evaluation used: `no`

## Command, seed, and environment

Benchmark command: pending

Public seed: `2040`

Reference SHA-256:
`0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172`

Evaluator SHA-256:
`0ab012597cfa79ec32ebc55bb28307c7b15309315f8057604760df5ad4be71db`

Pair order: odd reference-first, even candidate-first

Timeout: `300 seconds per evaluator process`

Measured region: evaluator timer around `run_solution(config)`

## Result

Not measured. Candidate editing is blocked until the survey and public
workload gates pass.

Decision: `baseline`

## Failure signal and interpretation

The first pinned-image build failed at Debian package index download after the
reported network interruption. This is an environment acquisition failure,
not evidence about the hypothesis.

## Next pivot

Complete the public Task 10 workload record, run reference profiling, retry
the pinned image build when network service returns, then commit e01 before
its first benchmark.

## Append-only corrections

None.

## Experiment `e01-exact-mps-scan`: measured result

Candidate commit: `05ba40c`

Candidate SHA-256:
`c1705d11d5c9e984ec7628b2dfcc08272c8ae3ff2e568c901de52f9a5ed1fc56`

Diff SHA-256:
`79d54f78de99c6b4242b43ef5a1809791bf9c686b836d29c4fe9de13fd991197`

Environment image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
(exploratory newer ORBIT-Q image, 6 CPUs / 7 GiB, network disabled).

The exact bond-2 CMZ MPO, bond-3 TFIM contraction, fused rotations, and
whole-training scan completed in 10.260367 seconds, but failed the functional
energy-gap gate:

```text
initial_energy_density: 0.9718867540
final_energy_density: -0.7872041464
exact_ground_energy_density: -1.2925285569
vqe_gap: 0.5053244105
Overall: FAIL
```

The first five reference and candidate histories agreed to low absolute error:

```text
reference: 0.9718865752, 0.8913648129, 0.7948441505, 0.6822248101, 0.5538985729
candidate: 0.9718867540, 0.8913646340, 0.7948451042, 0.6822254658, 0.5539006591
max parameter delta after five updates: 5.95897e-05
```

This establishes semantic equivalence at the start but numerical trajectory
divergence over 200 complex64 updates. The result is invalid and has no
performance standing.

Decision: `invalid`

### e01 numerical follow-ups

Commit `d9a5336` removed rotation fusion to reproduce the expert's explicit
`RX -> RZ -> RY` gate sequence. Its five-step maximum parameter delta grew to
`1.01790e-04`, so it was discarded as a regression.

Commit `3d48da0` restored fused rotations and changed only the MPS/CMZ/TFIM
intermediate dtype to complex128 while retaining float32 parameters and Adam
state. It completed `run_solution` in 10.790999 seconds but ended at
`-0.2665115864`, also invalid. Higher precision does not cure the optimization
basin sensitivity.

Failure interpretation: differentiating through exact QR/RQ canonicalization
is mathematically valid, but its gauge/rounding path perturbs this sensitive
200-step Adam trajectory. The next experiment will remove canonicalization
entirely: the known bond dimensions grow only `1 -> 2 -> 4`, so the raw
framework MPO-times-MPS tensors are already exact and bounded.

## Experiment `e02-noqr-mps`: promoted campaign best

Date: 2026-07-29

Branch: `codex/orbitbreakers/task-10/noqr-mps`

Latest accepted parent commit:
`02f4d37adadcdb33f7ba4336d69668d533378ffd`

Candidate commit:
`34171c020c2f9613efefaddefb94dc87ec9cbf16`

Candidate SHA-256:
`ad3154ccdfec1a329493e1dc7bbe6e3d30ee4e5d0cc7ac16c9922c57976d1262`

Expert-to-candidate diff SHA-256:
`8a404895e36a14246f82c4e40d239cb1b1ca2d12a042377b685953713011c06f`

Raw paired report SHA-256:
`a75f15f0cb67e93e27458e63248ffb462e4c7c8d7d3ee1c30ff7cdcbc8868f54`

Sanitized evidence:
[`profiles/e02-latest-nightly-paired.json`](profiles/e02-latest-nightly-paired.json)

### Hypothesis

The two exact bond-2 CMZ applications bound the state bond dimension by four.
Applying their local MPO-times-MPS contraction without QR/RQ will preserve the
expert's complex64 optimizer trajectory, shrink tracing/compilation work, and
pass the evaluator. The exact bond-3 TFIM expectation and all 200 Adam updates
remain in TensorCircuit-NG backend operations, with the updates executed by one
`K.jaxy_scan`.

Falsification rule: invalid if any evaluator gate fails, any exact rank is
truncated, fewer than 200 updates run, any returned shape changes, fewer than
five eligible matched pairs remain, the candidate loses two or more of the
first five pairs, or the predeclared paired-speedup 95% lower bound is at or
below 1.0.

### Data, command, seed, and environment

Public dataset version: `orbitq-workloads-v20260729.3`

Public Task 10 workload SHA-256:
`c978b7b0c45affa8c1842c0f26d19131a24c5ec8b7253a5e531ad48b7faa8340`

Private evaluation used: `no`

Seed: `2040`

Evaluator SHA-256:
`0ab012597cfa79ec32ebc55bb28307c7b15309315f8057604760df5ad4be71db`

Immutable reference SHA-256:
`0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172`

Command:

```bash
./bench run 10 \
  --solution optimized \
  --compare-to reference \
  --repeat 6 \
  --engine docker \
  --timeout 300 \
  --no-build \
  --output results/task-10-e02-latest-6cpu
```

Image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`

Image packages:
TensorCircuit nightly `1.8.0.dev20260726`, Python `3.11.15`,
JAX/JAXLIB `0.10.0`, Optax `0.2.8`, Quimb `1.11.1`, OMECo `0.2.4`,
TensorNetwork-NG `0.5.1`, NumPy `2.2.6`, SciPy `1.17.1`.

Container allocation: `6 CPUs`, `7 GiB`, network disabled.

Host fingerprint SHA-256:
`c627504db97dc65b8d998afb4b9cdf73cfc3eff2ce2ac589b4a7a4aa0c7fdc48`

Pair order: odd `reference -> candidate`, even
`candidate -> reference`.

Each cell started a fresh evaluator process in one shared container. The
measured region was the evaluator timer around `run_solution(config)`.

### Functional result

All six candidate cells passed. A direct full candidate evaluation reported:

```text
Initial energy density: 0.9718861580
Final energy density: -1.1781773567
Exact ground energy density: -1.2925285569
VQE energy-density gap: 0.1143512002
Energy history shape: (200,)
Final parameter shape: (2, 22, 3)
Overall: PASS
```

The immutable expert's profiled final energy density was `-1.1781759262`.
No singular-value truncation, rank cap, changed Hamiltonian, skipped update,
or hard-coded output is present.

### First-five result requested by the user

Pairwise runtimes `(reference, candidate, speedup)`:

```text
(20.254729, 3.908787, 5.181845)
(18.556855, 3.630935, 5.110765)
(18.895129, 4.107835, 4.599778)
(18.175488, 3.784015, 4.803228)
(18.774277, 3.914864, 4.795640)
```

```text
reference_mean_sec: 18.9312956
reference_median_sec: 18.774277
reference_stdev_sec: 0.7888107693
reference_stderr_sec: 0.3527669003
candidate_mean_sec: 3.8692872
candidate_median_sec: 3.908787
candidate_stdev_sec: 0.1765117922
candidate_stderr_sec: 0.0789384732
ratio_of_means_speedup: 4.8927088173
ratio_of_means_improvement_pct: 79.5614242060
paired_speedup_mean: 4.8982511930
paired_speedup_stderr: 0.1082202220
paired_speedup_ci_low: 4.5977836874
paired_speedup_ci_high: 5.1987186987
paired_wins: 5/5
```

### Supplemental sixth pair

Pair 6 measured `(18.869560, 3.665339, 5.148108)`. Across all six pairs:

```text
reference_mean_sec: 18.9210063333
candidate_mean_sec: 3.8352958333
ratio_of_means_speedup: 4.9333890150
ratio_of_means_improvement_pct: 79.7299585140
paired_speedup_mean: 4.9398939474
paired_speedup_stderr: 0.0976824615
paired_speedup_ci_low: 4.6887931860
paired_speedup_ci_high: 5.1909947088
paired_wins: 6/6
all_evaluator_cells_passed: yes
```

### Profiling result

The candidate profiling harness was added in later evidence-only commit
`fe82655`; it did not alter the measured candidate bytes.

Reference natural JIT boundary (one optimizer step):

```text
jit_lower_sec: 4.3702465110
xla_compile_sec: 13.1435420480
stablehlo_line_count: 20474
steady_step_ms: 0.3174230
full_run_solution_sec: 18.3647714800
```

Candidate natural JIT boundary (complete 200-step scan):

```text
jit_lower_sec: 0.6671965030
xla_compile_sec: 2.8734394700
stablehlo_line_count: 19252
steady_full_training_ms: 39.6777508
full_run_solution_after_profile_sec: 3.6463986430
```

Lowering plus XLA compilation fell from `17.513789 s` to `3.540636 s`,
a `79.78%` reduction. The natural entry signatures differ, so StableHLO line
counts are diagnostic; the cold-process paired evaluator timings remain the
performance authority.

### Decision

Decision: `accept-campaign-best-for-pr`

The candidate passes the functional contract, wins every pair, lowers mean
and median runtime, and has a paired-speedup confidence lower bound far above
1.0. It is eligible for a same-host relative improvement claim over the
bundled human expert.

The final comparison intentionally used the latest available TensorCircuit
nightly for both sides and a symmetric 6-CPU/7-GiB allocation because the
backend cannot supply the requested 8 CPUs. Per the user's direction, exact
downgrade to the repository's older TensorCircuit lock is not required. The
result is not labeled as the default pinned baseline or global SOTA.

No solution tuning occurred after this final paired benchmark. Subsequent
changes add only profiling, sanitized evidence, and research documentation.

## Append-only corrections

The initial campaign entry described the unavailable exact-lock image as a
prerequisite for the final comparison. The user later clarified that the
latest TensorCircuit release is acceptable and that only the same-machine
relative performance matters. Experiment e02 follows that clarified scope
and records the exact image ID and symmetric resource limits.

The publication branch was subsequently rebased onto upstream commit
`5af98f27b9404c513df8eee0f4568b1512edee19` after Task 11 PR #7 merged.
The measured candidate commit `34171c0` became
`a17d4237d37056de93272ba4ce3947493c1b7817`; its source SHA-256 remains
exactly `ad3154ccdfec1a329493e1dc7bbe6e3d30ee4e5d0cc7ac16c9922c57976d1262`,
and its expert-to-candidate diff SHA-256 remains exactly
`8a404895e36a14246f82c4e40d239cb1b1ca2d12a042377b685953713011c06f`.
The profiling-harness commit `fe82655` became `def9ba5`. No benchmarked code,
timing, or functional result changed.

The rebase resolved the additive public-manifest overlap by retaining the
Task 11 and Task 10 cases and advancing the current manifest to
`orbitq-workloads-v20260729.4`. The Task 10 canonical file and its SHA-256
remain unchanged; the benchmark-time dataset version recorded above remains
`orbitq-workloads-v20260729.3`.

Reviewer correction: the candidate module docstring is now restored
byte-for-byte from the immutable expert source. This changes the current
source SHA-256 to
`4696844a075c2fc18110f4d40e595c7c47bc69d58f2175bf28f8e76a80997998`.
The executable statements are unchanged, so the measured artifact and all
timing records correctly retain their original SHA-256 above.

## Post-publication starter-insight ablations

Date: 2026-07-29

The user supplied a starter-insight list after Draft PR #8 was opened. Four of
its five ordinary proposals were already present in e02: exact bond-2 CMZ MPO,
whole-training scan, fused rotations, and direct bond-3 TFIM MPO. OMECo budget
tuning was superseded because e02 removes OMECo and general path search from
the objective entirely.

The remaining specialized product-state-branch observation was tested in two
fresh experiment worktrees. With arbitrary local rotations between the two
CMZ reflections the exact general bound is four branches, not three:
one branch doubles to two after CMZ 1, local gates retain two, and CMZ 2 may
double both to four.

### Experiment `e03-product-branches`

Branch: `codex/orbitbreakers/task-10/product-branches`

Candidate commit: `c5bceb7`

Candidate SHA-256:
`2ab82c80b4df3841dde3400fac12570a32c794b1033e2049b2812be59ad140e8`

Sanitized evidence:
[`profiles/e03-five-step-comparison.json`](profiles/e03-five-step-comparison.json)

E03 represented the state as four explicit product branches and materialized
the 22 X and 21 ZZ expectation products from pairwise local overlaps. In one
shared, network-disabled container, five updates measured:

```text
e02 MPS:                 3.8043357890 s
e03 explicit branches: 16.0631762450 s
candidate / baseline:   4.2223339726
max history delta:      3.5762786865e-7
max parameter delta:    4.6938657761e-7
```

Decision: `invalid-performance`

The representation is exact at complex64 scale, but explicit Hamiltonian-term
products make the cold reverse-mode/XLA graph over four times slower.

### Experiment `e04-branch-transfer-scan`

Branch: `codex/orbitbreakers/task-10/branch-transfer`

Candidate commit: `a712df9`

Candidate SHA-256:
`79e866a12ddb00f438c36e51fa459ee13bfee62147cf326241f9ce8e8ff7b99e`

Sanitized evidence:
[`profiles/e04-five-step-screen.json`](profiles/e04-five-step-screen.json)

E04 retained the four branches but accumulated norm, all X insertions, all
adjacent ZZ insertions, and the previous Z insertion in one exact 4x4 transfer
scan. It reduced the five-step screen to `13.3214582620 s`, but remained
`3.5016515368x` slower than the fresh e02 screen. Its history stayed within
`4.7683715820e-7` of e02.

Decision: `invalid-performance`

Both experiments were stopped before the expensive full sparse-ground-state
evaluator because they falsified the predeclared early performance condition.
No accepted solution bytes or benchmark claim changed.

## Post-publication factor ablation

Date: 2026-07-29

The promoted e02 candidate bundled an exact bounded-rank MPS/MPO
representation, local rotation fusion, and a whole-training scan. Two
source-derived removal ablations isolate the execution-level factors while
leaving the structural quantum representation unchanged. The derivation
script asserts the exact source blocks before producing a temporary candidate.

Both comparisons used five counterbalanced pairs, fresh evaluator processes,
one network-disabled container, 6 CPUs, 7 GiB, and the same nightly image as
the primary campaign. All 20 cells passed.

```text
remove scan:
  promoted mean: 3.809923 s
  ablation mean: 3.752754 s
  paired ablation/promoted mean: 0.985715x
  95% t-CI: [0.939408, 1.032023]
  promoted wins: 2/5

remove local rotation fusion:
  promoted mean: 3.856303 s
  ablation mean: 4.051055 s
  paired ablation/promoted mean: 1.051484x
  95% t-CI: [0.975263, 1.127705]
  promoted wins: 5/5
```

Decision: retain rotation fusion as a small, directionally useful change.
Retain the scan for implementation compactness, but assign it no measured
speedup contribution. Attribute the dominant end-to-end gain to the coupled
low-rank MPS/CMZ/TFIM representation. Do not assign separate percentages to
representation-coupled subcomponents or treat QR/RQ removal as a
performance-only factor, because the QR/RQ trajectory fails correctness.

Artifacts:

- `run_factor_ablation.py`;
- `profiles/ablation-no-scan-five-pair.json`;
- `profiles/ablation-unfused-rotations-five-pair.json`.
