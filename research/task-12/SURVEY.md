# ORBIT-Q Task 12 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-12`

Survey freeze: `2026-07-28T13:12:11Z`

Reference commit: `690ffbac51715afd0a3e80718eeb6de20f11863a`

This survey covers only Task 12, as required by the one-task campaign scope in
`GOAL.md`. `READY` means the public knowledge and measurement plan are
complete. It does not attest that the public dataset or repeated-runtime gates
have passed.

## Provenance of this campaign's prior research

The bottleneck analysis, candidate design, and cross-validation summarized
below were first executed on 2026-07-28 against the byte-equivalent ORBIT-Q
publication reference (`optimized_sloutions/challenge-12` work in the ORBIT-Q
fork `QingyunQian/ORBIT-Q`, branch `cursor/optimize-challenge-12-f598`), on a
4 vCPU Intel Xeon cloud VM with `tensorcircuit-nightly==1.8.0.dev20260726`
and JAX 0.10.0. The two reference files differ only in suite naming
("Task Suite" vs "Challenge Suite") and documented evaluator timing precision.

Every profiling figure quoted in this survey was then **re-measured in this
repository's pinned lock environment** on the campaign host; the sanitized
profiler outputs live under `research/task-12/profiles/` with the scripts
beside them. Every runtime-improvement number for this repository comes from
fresh `./bench` paired runs recorded in [`LOG.md`](LOG.md). External-host
numbers are context only and are never pooled with in-repo measurements.

## Evidence and claim rules

Repository evidence below names a path, symbol or line, commit, and SHA-256.
Pinned-environment evidence was inspected in a virtual environment installed
exactly from `envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
External evidence links primary project documentation.

The bundled immutable human expert is the only reproducible performance
comparator currently available for this exact 32-qubit, 2-layer, 5000-update
overlap workload. The historical ORBIT-Q values in
`baselines/historical.json` (publication 6.12 s) and the shared-container
bootstrap (`baselines/bootstrap-2026-07-27.md`, 11.261 ± 0.972 s on 8 CPUs)
are context only. No cited external paper reports a matched evaluator
runtime, hardware allocation, software version, and output contract.
Therefore this campaign may claim a paired gain *over the bundled human
expert* when the frozen rule passes, but must not claim global state of the
art without a new matched external comparator.

## Pinned environment and inspected framework paths

The package versions come from `envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
The Dockerfile is `envs/tensorcircuit-py311/Dockerfile`
(`sha256:cbcffdfc0f17731bcdf549b94209879043aab8e53ee937be60947a49d2065912`).
The OMECo contractor shortcut backport is
`envs/tensorcircuit-py311/sitecustomize.py`
(`sha256:02800060761f2b15abe9055aded49d2af3877ab93d7b0fae8af94b30bac30120`);
it maps `tc.set_contractor("omeco")` to an `omeco.TreeSA(ntrials=16,
niters=32)` custom contractor with TensorCircuit preprocessing.

| Component | Pinned version | Inspected source or API |
| --- | --- | --- |
| TensorCircuit-NG | `tensorcircuit-nightly==1.7.0.dev20260618` | `tensorcircuit/gates.py:881`, `su4_gate` builds the generator as `sum(theta[i] * pauli_ops[i])` over 15 stacked Pauli pairs in the fixed order `ix, iy, iz, xi, xx, xy, xz, yi, yx, yy, yz, zi, zx, zy, zz`, then calls `backend.expm(-1j * generator)` per gate (`sha256:1c42762b1164c9bd33a28844ec3e42ab7eea5e853aecc5c4c741510becc5bf68`); `tensorcircuit/quantum.py:1525`, `quimb2qop`, and `quantum.py:889`, `QuVector` (`sha256:550e2cc7d41b4eadfe7ce6dd37f1d295539eb22a580f5e2993a0a2ec3408c1b4`); `tensorcircuit/quditcircuit.py:30,146`, `QuditCircuit` with `any(*indices, unitary=...)` accepting `(d, d)` or `(d^2, d^2)` unitaries (`sha256:bb765d5c33244c95a5b6cd7bfb4e15d7cdb1f6d99f991477f83acaab604fcf57`); `tensorcircuit/cons.py:1136`, `set_contractor` (`sha256:4fb517ad6085328c53ed61c75471579708eb8265fa6c14ec98e4bf1b4fd04365`) |
| JAX / JAXLIB | `0.10.0` | `tensorcircuit/backends/jax_backend.py:438`, `expm` delegates to `jax.scipy.linalg.expm`, the norm-adaptive Pade-13 scaling-and-squaring implementation (`sha256:22bbcf80d5a018884337286f176849ad4e8681b4e1f20efc1f80ba9ec4cc9d69`); `jax.lax.scan`, `jax.value_and_grad`, and `jax.jit` wrapped at `jax_backend.py:712,803,844` |
| OMECo | `0.2.4` | Registered through the sitecustomize `set_contractor("omeco")` shortcut; both the reference and every candidate select it at import time |
| TensorNetwork-NG | `0.5.1` | `QuOperator.eval` contracts the bra/ket node network with the configured contractor |
| Quimb | `1.11.1` | Supplies the evaluator-side DMRG MPS (`config["dmrg_state"]`); solutions only convert it (`quimb2qop` in the reference; pair fusion of `mps.arrays` in the tracked variant) |
| Optax | `0.2.8` | `optax.adam`, first/second-moment state on 465 parameters; the [Optax Adam API](https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html) documents the update rule the expert uses |

TensorCircuit documents JAX, automatic differentiation, JIT, vectorization,
and contractor selection as supported execution paths in its
[quick start](https://tensorcircuit.readthedocs.io/en/latest/quickstart.html)
and identifies contractor choice and preprocessing in the
[contractor tutorial](https://tensorcircuit.readthedocs.io/en/latest/tutorials/contractors.html).
JAX documents `lax.scan` as the loop primitive that compiles the body once
([jax.lax.scan API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)),
and `jax.scipy.linalg.expm` as a Pade-based matrix exponential
([jax.scipy.linalg.expm API](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.expm.html)).
The fixed-order diagonal Pade approximant used by the candidate follows the
classical scaling-and-squaring analysis (Higham, *The Scaling and Squaring
Method for the Matrix Exponential Revisited*, SIAM J. Matrix Anal. Appl.
26(4), 2005, <https://doi.org/10.1137/04061101X>); diagonal Pade approximants
of anti-Hermitian arguments are exactly unitary, a standard property of
Cayley-type transforms used in geometric integration (Hairer, Lubich, Wanner,
*Geometric Numerical Integration*, ch. IV).

## Task 12: variational circuit to DMRG-MPS overlap optimization

### Immutable expert and output contract

The immutable expert is `references/task-12/solution_12.py`
(`sha256:10cfd516bc250633f4675653e0d8986002e56f4d5916a9c2972c1085193f5d38`).
The contract and timer are `tasks/task-12/problem.md`
(`sha256:cd795bbbd5fcf2beb0000528e1e169fa09a4f20bdbc829381b2bbf7810c36fec`)
and `tasks/task-12/evaluator/evaluate_12.py`
(`sha256:08940a5fabfd88a957c467edabfbe6faa7b766f38b4d518557e50e94fcf3b277`).

The evaluator builds a chi-8 DMRG MPS target for a 32-qubit XXZ chain with a
staggered field **before** timing, passes it in as `config["dmrg_state"]`,
and times only `run_solution(config)`. The solution must:

- prepare the Neel product state `|0101...01>` on 32 qubits;
- apply two brickwork layers of trainable nearest-neighbor SU4 gates (even
  bonds then odd bonds; 16 + 15 = 31 gates, 15 su(4) angles each, 465
  parameters), where the expert's SU4 convention is
  `tc.gates.su4_gate`: `expm(-1j * sum_i theta_i G_i)` over the fixed
  15-generator Pauli-pair order;
- maximize `F = |<psi_MPS | psi_circuit(theta)>|^2` through the direct
  tensor-network overlap of the MPS bra with the circuit ket (no
  gate-preparation conversion of the target);
- run exactly 5000 Adam updates at learning rate 0.02 from the seeded
  `float32` initialization (`numpy.random.default_rng(2039)`, scale 0.02),
  recording pre-update loss and fidelity per step;
- return `loss_history (5000,)`, `fidelity_history (5000,)`,
  `final_parameters (465,)`, and `final_overlap_phase` as NumPy data, with
  final fidelity at least 0.85, improving fidelity/loss, and finite values.

### Dominant work and measured bottlenecks

Stage-split profiling of the reference in the pinned lock environment on the
campaign host (`research/task-12/profile_reference.py`, AOT-split via
`jit(...).lower()` / `.compile()`; sanitized output at
`research/task-12/profiles/reference-profile.json`) shows the reference's
end-to-end time is dominated by graph construction and compilation, not by
tensor-network arithmetic:

- one-time `quimb2qop` conversion: ~0.14 s;
- one-time jit trace (Python tracing of 31 per-gate `su4` constructions plus
  the TreeSA contraction-path search): ~1.7 s;
- one-time XLA compile of an 8884-line StableHLO module: ~3.4 s;
- 5000 sequential Adam steps at ~0.67 ms per step: ~3.3 s.

Inside the step, gate construction dominates: a scan step containing only the
batched gate build costs ~0.44 ms versus ~0.45 ms for a full fused step
(external-host decomposition, reproduced in spirit by
`profiles/expm-microbench.json` in the pinned environment), because each
`tc.gates.su4_gate` call chains 15 scalar multiply/stack operations and one
norm-adaptive Pade-13 `jax.scipy.linalg.expm` (norm estimate, conditionals,
LU solves) per gate, and the whole chain is differentiated. The contraction
itself is nearly free: exactly one brickwork gate crosses each bipartition
cut, so the exact circuit state has bond dimension at most 4 and every
network tensor is tiny.

### Relevant TensorCircuit-NG primitives and contraction paths

- `tc.Circuit(n)` with `circuit.any(i, i+1, unitary=...)` applies a
  precomputed `(4, 4)` unitary exactly like `circuit.su4` applies its
  internally built matrix (`gates.py:881`), so batched gate-matrix
  construction preserves gate semantics bit-for-bit up to float ordering.
- `tc.quantum.quimb2qop` wraps the quimb MPS arrays as a `QuOperator` bra;
  `(target_bra @ circuit.quvector()).eval()` contracts the joint network with
  the globally configured contractor (sitecustomize TreeSA `omeco` shortcut).
- `tc.QuditCircuit(16, dim=4)` supports single- and two-site `(d, d)` /
  `(d^2, d^2)` unitaries and `quvector()`, enabling an exact pair-fused
  formulation (tracked as the `fused` variant).
- `tc.backend.expm` is a thin alias of `jax.scipy.linalg.expm`; nothing in
  the framework requires the norm-adaptive path for fixed-size 4x4
  anti-Hermitian generators.

### Candidate optimization hypotheses and semantic constraints

Each hypothesis must preserve: the su(4) generator order and
`expm(-1j * sum theta G)` gate semantics, the seeded float32 initialization,
exactly 5000 sequential Adam updates at learning rate 0.02, pre-update
loss/fidelity recording, the direct MPS-bra x circuit-ket overlap (no target
state preparation circuit, no oracle shortcuts), complex64 TensorCircuit/JAX
semantics, and the full output contract.

1. **e01 (primary): batched fixed-order su(4) exponentials plus
   whole-training scan.** Build all 31 generators with one einsum against the
   stacked generator tensor; exponentiate with a fixed 2^5
   scaling-and-squaring diagonal Pade(3,3) core (exactly unitary for
   anti-Hermitian input; error below the complex64 noise floor for the
   observed parameter range, margin documented in
   `profiles/expm-microbench.json`); apply through `circuit.any`; run the
   5000 updates in one `jax.lax.scan`. Expected effect: order-of-magnitude
   StableHLO shrinkage, several-fold lower trace/compile time, several-fold
   lower per-step time.
2. **Pair-fused ququart contraction (tracked variant).** Fuse qubit pairs
   into 16 four-level sites (`QuditCircuit(16, dim=4)`); layer-1 gates become
   single-site unitaries with the Neel preparation folded in as a constant
   basis permutation; layer-2 gates become two-site `I2 (x) SU4 (x) I2`
   unitaries; the target MPS is pair-fused once outside the loop. Halves the
   contraction network without changing any contraction result.
3. **Scan alone (rejected by prior evidence).** On the unmodified reference
   objective the scan converts ~5 us/step dispatch and nothing else;
   external-host measurement saw ~9% end to end. Not pursued separately.
4. **Coarser fusion (rejected by prior evidence).** Fusing 4 qubits per site
   (dim-16 crossing gates of shape 256x256) inflates contraction FLOPs and
   was measured ~4x slower than the reference formulation externally. Do not
   repeat unchanged.

### External comparison state

No published external system reports runtime for this exact workload
(32-qubit XXZ chi-8 DMRG target, 465-parameter SU4 brickwork, 5000 Adam
updates, this evaluator timer, this CPU allocation). The strongest available
comparators are the bundled immutable expert (this repository) and the
ORBIT-Q publication runtime record (6.12 s, unspecified publication host,
`baselines/historical.json`). Open evidence gap: no matched-hardware external
baseline exists; global SOTA cannot be claimed.

## Frozen paired measurement and statistics plan

This section is frozen before candidate timing results.

- Workload: public dataset version recorded in
  `datasets/public/manifest.json`; Task 12 canonical fixed configuration
  only.
- Engine/allocation: this campaign host has no Docker daemon available, so
  eligible in-repo measurements use `./bench run 12 ... --engine local` with
  a virtual environment installed exactly from
  `envs/tensorcircuit-py311/requirements.lock` (Python 3.12.3), the
  environment `sitecustomize.py` on `PYTHONPATH`, `NUMBA_DISABLE_JIT=1`, and
  a fresh evaluator process per cell on an otherwise idle 4 vCPU host. The
  Docker single-container protocol of `GOAL.md` Gate 3 therefore remains
  **closed** on this host; the Docker-gated promotion claim is deferred to a
  maintainer or CI rerun with `--engine docker`.
- Cells: six matched pairs minimum; a fresh evaluator process per cell.
- Order: odd pairs `reference -> candidate`; even pairs
  `candidate -> reference` (the `./bench` counterbalanced default).
- Timeout: hard 300 seconds per evaluator process.
- Cache state: cold process-local Python/JAX caches for every fresh process;
  no persistent compilation cache; no candidate-only precomputation outside
  `run_solution`.
- Validity: process exit zero, finite positive evaluator runtime,
  `Overall: PASS`, full output schema, exactly 5000-step histories.

For eligible runtimes `R_i` and `C_i`, report means, medians, sample standard
deviations, standard errors, minima, maxima, ratio-of-means improvement
`100*(mean(R)-mean(C))/mean(R)`, pairwise improvements `100*(R_i-C_i)/R_i`,
and pairwise speedups `S_i=R_i/C_i`.

The predeclared confidence interval is the two-sided 95% Student-t interval
for the arithmetic mean of the pairwise speedups:

`mean(S) ± t_(0.975,n-1) * sample_stdev(S)/sqrt(n)`.

With six pairs, the critical value is `t_(0.975,5)=2.5705818366`. Do not
change the method after observing a candidate.

A candidate is promotable only when every validity gate passes, at least six
pairs are eligible, candidate mean and median are lower, the candidate wins
at least 80% of pairs, and the 95% lower confidence bound for mean paired
speedup exceeds 1.0 -- and, for the repository's formal Docker promotion
gate, when the same protocol is reproduced under `--engine docker` per
`GOAL.md` Gate 3. Until that Docker rerun exists, results below are recorded
as local-engine paired evidence, not a promoted Docker-gated claim.

## Open evidence gaps

- No Docker daemon on the campaign host: Gate 3 reference baselines and the
  formal promotion protocol cannot run here; local-engine paired evidence is
  recorded instead and a Docker rerun is requested in the campaign report.
- No matched external implementation publishes runtime for this exact
  configuration; global SOTA remains unestablished.
- Process peak RSS was not measured; XLA memory-traffic estimates were not
  collected in the pinned environment.
