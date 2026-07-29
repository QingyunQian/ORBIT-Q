# Task 10 Human-Expert Optimization Report

## Scope and claim

This campaign optimized only ORBIT-Q Task 10: a 22-qubit, two-layer VQE with
an 18-qubit controlled-Z hyperedge and exactly 200 Adam updates. The immutable
human expert and the optimized candidate were evaluated with the unchanged
Task 10 evaluator, canonical public configuration, and seed.

The candidate is the **campaign-best implementation**. It is not described as
a global SOTA implementation because no matched external implementation
publishes a runtime for this exact workload, evaluator, TensorCircuit version,
and CPU allocation.

The user-requested five-pair comparison establishes a `4.8927x`
ratio-of-means speedup and a `79.56%` end-to-end runtime reduction. A sixth
predeclared pair was run as additional repository-grade statistical evidence;
all 12 evaluator cells passed and the candidate won all six pairs.

| Role | Artifact | Commit or hash |
|---|---|---|
| Immutable human expert | [`references/task-10/solution_10.py`](../../../tasks/challenge-10/solution/solution_10.py) | SHA-256 `0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172` |
| Campaign best | [`src/solutions/task-10/solution_10.py`](../solution_10_exact_mps.py) | Current review-corrected SHA-256 `4696844a075c2fc18110f4d40e595c7c47bc69d58f2175bf28f8e76a80997998`; measured pre-rebase commit `34171c020c2f9613efefaddefb94dc87ec9cbf16` and measured SHA-256 `ad3154ccdfec1a329493e1dc7bbe6e3d30ee4e5d0cc7ac16c9922c57976d1262` |
| Candidate diff | Expert-to-candidate patch | SHA-256 `8a404895e36a14246f82c4e40d239cb1b1ca2d12a042377b685953713011c06f` |
| Evaluator | [`tasks/task-10/evaluator/evaluate_10.py`](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/blob/7e2298b/tasks/task-10/evaluator/evaluate_10.py) | SHA-256 `0ab012597cfa79ec32ebc55bb28307c7b15309315f8057604760df5ad4be71db` |
| Sanitized measurement | [`profiles/e02-latest-nightly-paired.json`](profiles/e02-latest-nightly-paired.json) | Raw report SHA-256 `a75f15f0cb67e93e27458e63248ffb462e4c7c8d7d3ee1c30ff7cdcbc8868f54` |
| Public workload | [`datasets/public/task-10/canonical.json`](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/blob/7e2298b/datasets/public/task-10/canonical.json) | SHA-256 `c978b7b0c45affa8c1842c0f26d19131a24c5ec8b7253a5e531ad48b7faa8340` |

## Benchmark result

Both implementations ran in one long-lived, network-disabled Docker
container. Every cell used a fresh evaluator process. Pair order alternated
to reduce drift:
`reference -> candidate`, then `candidate -> reference`.

The main result below uses the first five pairs, as requested.

| Metric | Immutable expert | Campaign best |
|---|---:|---:|
| Passing runs | 5/5 | 5/5 |
| Mean runtime | 18.931296 s | 3.869287 s |
| Median runtime | 18.774277 s | 3.908787 s |
| Sample standard deviation | 0.788811 s | 0.176512 s |
| Standard error | 0.352767 s | 0.078938 s |
| Ratio-of-means speedup | — | **4.892709x** |
| Ratio-of-means runtime reduction | — | **79.561424%** |
| Mean paired speedup | — | **4.898251x** |
| Paired-speedup standard error | — | 0.108220x |
| Paired-speedup 95% Student-t interval | — | 4.597784x–5.198719x |

Matched timings:

| Pair | Order | Expert | Candidate | Speedup |
|---:|---|---:|---:|---:|
| 1 | reference → candidate | 20.254729 s | 3.908787 s | 5.181845x |
| 2 | candidate → reference | 18.556855 s | 3.630935 s | 5.110765x |
| 3 | reference → candidate | 18.895129 s | 4.107835 s | 4.599778x |
| 4 | candidate → reference | 18.175488 s | 3.784015 s | 4.803228x |
| 5 | reference → candidate | 18.774277 s | 3.914864 s | 4.795640x |

The sixth supplemental pair measured `18.869560 s` versus `3.665339 s`
(`5.148108x`). Across all six pairs, expert and candidate means were
`18.921006 s` and `3.835296 s`; the ratio-of-means speedup was `4.933389x`,
the mean paired speedup was `4.939894x`, and its 95% Student-t interval was
`4.688793x–5.190995x`.

## Optimization strategy

### Exact bounded-rank MPS state

Single-qubit rotations do not increase MPS bond dimension. TensorCircuit-NG's
CMZ is an exact bond-2 diagonal tensor chain, so two CMZ applications grow
the candidate's exact state bonds only as:

```text
product state: 1
after CMZ 1:   2
after CMZ 2:   4
```

The candidate therefore represents the state with `tc.MPSCircuit` tensors
instead of rebuilding a general circuit-bra/MPO/circuit-ket network for every
energy call.

### Exact CMZ without unnecessary canonicalization

The CMZ action is:

```text
I - 2 * product(selected sites |1><1|)
```

The candidate stores this as a bond-2 MPO and applies the local contraction
kernel used by TensorCircuit-NG's `MPSCircuit.apply_MPO`:

```python
contracted = K.einsum("iabj,kbl->ikajl", operator, tensor)
```

It intentionally omits the method's QR/RQ sweep. No compression is needed:
the exact rank bound is four, so canonicalization adds work without reducing
the state. This was also necessary numerically. The discarded `e01`
implementation used framework QR/RQ canonicalization and matched the first
updates closely, but its round-off/gauge path sent the sensitive 200-step
Adam trajectory to an evaluator-invalid basin.

### Deterministic bond-3 TFIM expectation

The open-boundary Hamiltonian
`-sum(Z_i Z_(i+1)) - 1.05 sum(X_i)` is represented as an exact bond-3 MPO.
The expectation is a fixed left-to-right TensorCircuit backend contraction
over bonds no larger than four. This removes OMECo path search from the traced
objective while preserving every Hamiltonian term.

### TensorCircuit-native gates and backend operations

The candidate builds `RX`, `RZ`, and `RY` with
`tc.gates.rx/rz/ry`, composes their ordered 2x2 unitary, and applies it with
`tc.backend.einsum`. MPS ownership remains with `tc.MPSCircuit`; differentiation,
JIT compilation, tensor conversion, einsums, and the scan use TensorCircuit's
selected JAX backend.

### Whole-training scan

The expert JIT-compiles one optimizer step and dispatches it from Python 200
times. The candidate places all 200 sequential Adam updates in one
`K.jaxy_scan`, returning the required pre-update energy history as the scan
output. This preserves update order and removes Python dispatch from the
timed training loop.

## Correctness and preserved semantics

All six candidate evaluator runs passed:

- 22 qubits, two ansatz layers, and the canonical 18 CMZ sites;
- identical seeded float32 parameter initialization;
- ordered `RX -> RZ -> RY` local rotations and one exact CMZ per layer;
- identical open-boundary TFIM Hamiltonian and energy-density normalization;
- exactly 200 sequential Adam updates at learning rate `0.03`;
- pre-update history shape `(200,)`;
- final parameter shape `(2, 22, 3)`;
- finite NumPy outputs and the evaluator's variational/gap thresholds.

The final candidate validation reported:

```text
Initial energy density:      0.9718861580
Final energy density:       -1.1781773567
Exact ground energy density:-1.2925285569
VQE energy-density gap:      0.1143512002
Overall: PASS
```

The immutable expert's profiled final energy was `-1.1781759262`. The
`1.43e-6` difference is ordinary complex64 optimizer-trajectory round-off;
both are comfortably inside the unchanged `0.25` gap threshold.

## Profiling evidence

The immutable expert is dominated by graph staging, OMECo path work during
tracing, and XLA compilation. Its compiled numerical step is already fast.
The candidate replaces the generic network with fixed, low-rank contractions.

| Phase | Immutable expert | Campaign best |
|---|---:|---:|
| JAX lowering | 4.370247 s | 0.667197 s |
| XLA compilation | 13.143542 s | 2.873439 s |
| Lowering + compilation | 17.513789 s | 3.540636 s |
| StableHLO lines | 20,474 | 19,252 |
| Compiled execution | 0.317423 ms per expert step | 39.677751 ms per complete 200-step scan |
| Full `run_solution` | 18.364771 s | 3.646399 s after profiling |

The staging-plus-compilation total fell by `79.78%` (`4.95x`). The two
profilers deliberately stage the implementations at their natural JIT
boundary: one expert optimizer step versus the candidate's full scan. The
StableHLO sizes are therefore diagnostic, not a claim that the modules expose
identical entry signatures. The repeated cold-process evaluator benchmark is
the authoritative end-to-end result.

Profilers and sanitized outputs:

- [`profile_reference.py`](profile_reference.py) /
  [`profiles/reference-profile.json`](profiles/reference-profile.json)
- [`profile_candidate.py`](profile_candidate.py) /
  [`profiles/candidate-profile.json`](profiles/candidate-profile.json)

## Factor ablation

The primary `4.8927x` result combines a structural representation change with
two smaller execution choices. To avoid assigning the full gain to every item
in that list, two removal ablations were measured after publication. Each row
uses five counterbalanced pairs in one network-disabled 6-CPU/7-GiB container,
a fresh unchanged evaluator process per cell, and the promoted candidate as
the control. All 20 cells passed.

| Removed factor | Promoted mean | Ablation mean | Paired ablation / promoted | Promoted wins | Interpretation |
|---|---:|---:|---:|---:|---|
| Whole-training `K.jaxy_scan` | 3.809923 s | 3.752754 s | 0.9857x, 95% t-CI `[0.9394, 1.0320]` | 2/5 | No positive contribution is resolved; the Python-dispatch loop is slightly faster on mean. |
| Fused `RX -> RZ -> RY` application | 3.856303 s | 4.051055 s | 1.0515x, 95% t-CI `[0.9753, 1.1277]` | 5/5 | Directionally useful but small: about 4.8% lower mean runtime, not a separately significant headline result. |

The conclusion is therefore narrower than the original strategy list:

- the whole-training scan is **not** responsible for the reported speedup;
- local rotation fusion is a small secondary factor;
- the dominant factor is the exact bounded-rank MPS/MPO representation that
  removes the expert's generic circuit-bra/MPO/circuit-ket path search and
  cold compilation graph. Even the slower unfused ablation remains about
  `18.931296 / 4.051055 = 4.67x` faster than the primary expert mean.

The structural core groups the bond-2 CMZ application, bond-3 TFIM
expectation, and low-rank MPS state. Those pieces are representation-coupled:
removing the MPS representation also removes the fixed local MPO contraction
being measured. The report therefore does **not** invent independent
percentages for those inseparable subcomponents. QR/RQ removal is likewise a
correctness requirement in this float32 optimizer trajectory, not a clean
performance-only factor: the corresponding framework-canonicalized candidate
failed the unchanged evaluator.

Reproduction and sanitized measurements:

- [`run_factor_ablation.py`](run_factor_ablation.py)
- [`profiles/ablation-no-scan-five-pair.json`](profiles/ablation-no-scan-five-pair.json)
- [`profiles/ablation-unfused-rotations-five-pair.json`](profiles/ablation-unfused-rotations-five-pair.json)

![Task 10 five-pair factor-ablation plots](figures/factor-ablation.svg)

Each line is one matched pair; pair numbers are labeled at the ablation end.
The figure is generated directly from the two tracked JSON records with
[`plot_factor_ablation.py`](plot_factor_ablation.py).

## Environment and reproducibility

The final paired campaign used the latest TensorCircuit nightly image already
available on the host, as permitted for this optimization:

```text
image ID: sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833
TensorCircuit nightly: 1.8.0.dev20260726
Python: 3.11.15
JAX/JAXLIB: 0.10.0
Optax: 0.2.8
container limit: 6 CPUs, 7 GiB
network: disabled
timeout: 300 seconds per evaluator process
```

The repository default asks for 8 CPUs, but the Docker backend is capped at
6. This does not invalidate the relative claim: reference and candidate were
staged before one shared container started, received the same 6-CPU/7-GiB
limits, and alternated execution order. These results are not pooled with
another hardware profile.

The measurement command was:

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

The full raw report is intentionally untracked because it contains
machine-local paths and container identifiers. Its SHA-256 and all sanitized
timings, hashes, environment facts, and statistics are preserved in
[`profiles/e02-latest-nightly-paired.json`](profiles/e02-latest-nightly-paired.json).

## Failed experiment retained

Experiment `e01-exact-mps-scan` used the same exact bond-2 CMZ and bond-3 TFIM
ideas but called `MPSCircuit.apply_MPO`, including QR/RQ canonicalization.
It completed in `10.260367 s` but ended at energy density `-0.7872041464`,
outside the evaluator threshold. Rotation-order and complex128 follow-ups
also failed. These are recorded in [`LOG.md`](LOG.md) and receive no speedup
claim.

## Starter-insight coverage and follow-up

| Starter insight | Used in e02? | Outcome |
|---|---|---|
| Native exact CMZ MPO | Yes | Exact bond-2 `I - 2P` representation |
| Tune OMECo or greedy | Superseded | E02 removes generic contraction/path search entirely |
| One scan for 200 Adam steps | Yes | `K.jaxy_scan` around the complete optimizer |
| Fuse `RX -> RZ -> RY` | Yes | One TensorCircuit-built 2x2 unitary per qubit/layer |
| Direct TFIM MPO | Yes | Exact bond-3 MPO, no Quimb conversion |
| Few product-state branches | Follow-up | Exact but slower in both tested forms |

The last observation requires up to four branches in the general two-layer
workload, rather than three, because local rotations separate the two CMZ
reflections. Two TensorCircuit-backend prototypes preserved the five-step
trajectory within `4.77e-7`:

- explicit branch-pair Hamiltonian products: `16.063 s`;
- one compact 4x4 branch transfer scan: `13.321 s`;
- accepted e02 MPS screen: `3.804 s`.

The branch transfer was a meaningful improvement over explicit term products,
but still a `3.50x` regression. Both were rejected before full evaluation;
details and source hashes are preserved in [`LOG.md`](LOG.md) and
[`profiles/e03-five-step-comparison.json`](profiles/e03-five-step-comparison.json) /
[`profiles/e04-five-step-screen.json`](profiles/e04-five-step-screen.json).

## Claim limits and next work

- The result establishes a same-host, same-container improvement over the
  bundled human expert for the one canonical public Task 10 workload.
- It does not establish global SOTA, cross-hardware performance, scaling, or
  an advantage on a different layer count or optimizer schedule.
- The final comparison uses TensorCircuit nightly
  `1.8.0.dev20260726`, not the repository's older exact lock. The code uses
  public TensorCircuit backend, gate, and MPS APIs, but this campaign does not
  claim a separately measured result under every TensorCircuit release.
- The `10x` research stretch target was not reached. Reaching it would require
  reducing the remaining roughly 3.5 seconds of cold lowering/compilation;
  compiled execution itself is already about 40 ms for all 200 updates.
