# ORBIT-Q Task 11 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-11`

Survey freeze: `2026-07-28T14:55:00Z`

Reference commit: `ed382bf042ecb1c87b399acaadec6bce74368649`

This survey covers only Task 11, as required by the one-task campaign scope
in `GOAL.md`. `READY` means the public knowledge and measurement plan are
complete. It does not attest that the public dataset or repeated-runtime
gates have passed.

## Campaign selection and provenance of prior research

Live open pull requests inspected on 2026-07-28: on `sxzgroup/ORBIT-Q`,
`#2` (ForgeCode agent), `#3` (scoring fix), `#4` (Fable 5 run record), `#5`
(GPT-5.6 Sol results); on this repository, `#4` (Task 01 MPO energy). None
is an active Task 11 solution-improvement PR, so Task 11 is eligible.

The bottleneck analysis, candidate design, and cross-validation summarized
below were first executed on 2026-07-28 against the byte-equivalent ORBIT-Q
publication reference in the fork `QingyunQian/ORBIT-Q` (branch
`cursor/optimize-challenge-11-f598`, `optimized_sloutions/challenge-11/`,
PR #5 there), on the same 4 vCPU cloud VM with
`tensorcircuit-nightly==1.8.0.dev20260726`. Every profiling figure quoted
below was then **re-measured in this repository's pinned lock environment**
(sanitized outputs under `research/task-11/profiles/`, scripts beside them).
All runtime claims for this repository come from fresh `./bench` paired runs
recorded in [`LOG.md`](LOG.md); external-host numbers are context only and
are never pooled with in-repo measurements.

## Evidence and claim rules

Repository evidence below names a path, symbol or line, commit, and SHA-256.
Pinned-environment evidence was inspected in a virtual environment installed
exactly from `envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).

The bundled immutable human expert is the only reproducible performance
comparator for this exact 12-site spin-1, 5-layer, 500-update workload with
this evaluator timer. The shared-container bootstrap measured it at
153.612 ± 5.039 s on the maintainer's 8-CPU Docker host
(`baselines/bootstrap-2026-07-27.md`); the ORBIT-Q publication record lists
68.10 s on an unspecified host (`baselines/historical.json`). No cited
external paper reports a matched runtime. This campaign may claim a paired
gain over the bundled expert when the frozen rule passes, but not a global
state of the art.

## Pinned environment and inspected framework paths

Package versions come from `envs/tensorcircuit-py311/requirements.lock`;
the OMECo shortcut backport `envs/tensorcircuit-py311/sitecustomize.py`
(`sha256:02800060761f2b15abe9055aded49d2af3877ab93d7b0fae8af94b30bac30120`)
is on `PYTHONPATH` for every measurement but is not exercised by Task 11
(neither the reference nor the candidate selects a contractor).

| Component | Pinned version | Inspected source or API |
| --- | --- | --- |
| TensorCircuit-NG | `tensorcircuit-nightly==1.7.0.dev20260618` | `tensorcircuit/quditcircuit.py:30`, `QuditCircuit` with dense `inputs=`, `unitary` (`any`, line 146) and `expectation` (line 417) (`sha256:bb765d5c33244c95a5b6cd7bfb4e15d7cdb1f6d99f991477f83acaab604fcf57`); backend `kron`/`expm`/`scan` wrappers at `tensorcircuit/backends/jax_backend.py:352,438,712` (`sha256:22bbcf80d5a018884337286f176849ad4e8681b4e1f20efc1f80ba9ec4cc9d69`) |
| JAX / JAXLIB | `0.10.0` | `jax.lax.scan` for the layer scan and update loop; `jax.value_and_grad`; XLA CPU lowering of strided-axis einsums to transpose+gemm (measured, see below) |
| Optax | `0.2.8` | `optax.adam` on the 7-leaf parameter tree |
| Quimb / OMECo / TensorNetwork-NG | `1.11.1` / `0.2.4` / `0.5.1` | Installed support libraries; not exercised by the Task 11 expert or candidate |

## Task 11: spin-1 Haldane-chain VQE with string-order readout

### Immutable expert and output contract

The immutable expert is `references/task-11/solution_11.py`
(`sha256:087c7a2894b4f0383bfc476f835933940cdfd2d9812f814adede3a39375b3f00`).
The contract and timer are `tasks/task-11/problem.md`
(`sha256:06f57943894e0455fcc5935216bcb0d3706628a38f81c5c7143ec5b95104ad78`)
and `tasks/task-11/evaluator/evaluate_11.py`
(`sha256:de70880ec00a86a7123aed14651b33401a7f872f667fb1598bd3ba191e29353b`).

The solution must, on 12 three-level sites (dense 3^12 complex64 state):

- start from the spin-1 Neel product state `|0,2,0,2,...>`;
- apply 5 brickwork layers, each with per-site rz/ry/rz spin-1 rotations
  followed by even-bond and odd-bond entanglers
  `expm(-i[theta S.S + (phi - theta) SzSz + beta (S.S)^2])` (7 stacked
  parameter leaves, seeded float32 initialization from
  `default_rng(2041)`, scale 0.05);
- minimize the energy density of the spin-1 chain
  `H = sum_bonds [S.S + 0.2 (S.S)^2] + 0.15 sum_i (S_i^z)^2` with exactly
  500 Adam updates at learning rate 0.03, recording the pre-update energy
  density per step;
- return the 500-entry history, the final energy density, and the three
  `<S^z_i exp(i pi sum S^z) S^z_j>` string-order values;
- pass: energy improvement >= 5e-3, gap to the exact sparse ground state
  <= 0.12, string-order MAE <= 0.12, finite arrays.

### Dominant work and measured bottlenecks

Stage-split profiling in the pinned lock environment
(`research/task-11/profile_reference.py`,
`profiles/reference-profile.json`): jit trace 1.32 s + XLA compile 2.73 s
(8135-line StableHLO), then a 500-step loop at 323 ms per step - the loop is
~97% of end-to-end time, the opposite regime from Task 12. Forward
`build_state` costs 81 ms (5 layers x 47 dense-state gate applications:
36 single-site + 11 two-site per layer), forward energy 36 ms (23 separate
`expectation` contractions), and the backward pass roughly doubles the rest.

The per-gate cost is memory-bandwidth-bound and position-dependent
(`profiles/gate-application-microbench.json`): one 9x9 two-qudit contraction
against the 4.25 MB dense state costs 0.5-2.9 ms depending on the
bond position, reflecting XLA's transpose+gemm lowering of strided-axis
einsums rather than FLOPs. A bare reshape-matmul floor probe shows only
~1.7x headroom below the framework circuit path and is recorded as a
boundary datapoint only (hand-rolled evolution is not shippable under the
framework-fidelity rules).

### Relevant TensorCircuit-NG primitives

- `tc.QuditCircuit(n, dim=3, inputs=state)` with `unitary` applies dense
  two-qudit gates; `circuit.state()` materializes the evolved state.
- `circuit.expectation((gate, [sites...]))` evaluates one operator term per
  call; product strings (used for the string order) are one call.
- `K.scan` stages repeated layers; `K.kron`, `K.expm`, `K.value_and_grad`
  wrap the JAX primitives.

### Candidate optimization hypotheses and semantic constraints

Every hypothesis must preserve: the parameter layout and seeded
initialization, the layer structure and gate semantics (per-site rz/ry/rz
then even/odd entanglers with the exact generator), exactly 500 sequential
Adam updates at learning rate 0.03 with pre-update recording, the energy
density and string-order definitions, complex64 semantics, dense-state
evolution through `tc.QuditCircuit`, and the output contract.

1. **e01 (primary): exact gate fusion + batched fixed-order entangler
   exponentials + diagonal-observable coefficient vector + whole-training
   scan.** Compose the three single-site rotations per site into one 3x3
   unitary (batched over sites), absorb each pair's composite into its
   even-bond 9x9 entangler (even bonds tile all sites), exponentiate all
   generators of a layer in one batched fixed 2^5 scaling-and-squaring
   diagonal Pade(3,3) pass, evaluate the diagonal single-ion term with one
   precomputed per-basis-state coefficient vector, run the 500 updates in
   one `jax.lax.scan`, and jit the post-training readout. Layer unitaries
   are algebraically identical; state passes drop 47 -> 11 per layer.
2. **Layer unrolling instead of `K.scan` (rejected by prior evidence):**
   295.5 vs 264.6 ms per step externally; larger compile, worse scheduling.
3. **Splitting even/odd into separate circuits (rejected):** 285.3 ms per
   step externally; one extra materialization per layer.
4. **Fusing 4 sites into dim-81 blocks (rejected on FLOPs):** 81x81
   crossing gates turn a bandwidth-bound problem into a compute-bound one
   (analogous to the Task 12 dim-16 result).

### External comparison state

No published external system reports runtime for this exact workload and
evaluator. Open evidence gap: no matched-hardware external baseline; global
SOTA cannot be claimed.

## Frozen paired measurement and statistics plan

Identical to the Task 12 campaign plan
(`research/task-12/SURVEY.md`), with the same engine deviation: this
campaign host has no Docker daemon, so eligible in-repo measurements use
`./bench run 11 ... --engine local` with the lock-pinned virtual
environment (Python 3.12.3), the environment `sitecustomize.py` on
`PYTHONPATH`, `NUMBA_DISABLE_JIT=1`, a fresh evaluator process per cell,
counterbalanced pair order (odd pairs `reference -> candidate`, even pairs
`candidate -> reference`), a hard 300-second cap per cell, cold caches, and
at least six matched pairs. The predeclared confidence rule is the
two-sided 95% Student-t interval on the mean pairwise speedup
(`t_(0.975,5)=2.5705818366` for six pairs); promotion additionally requires
every validity gate, lower candidate mean and median, at least 80% of pairs
won, and a CI lower bound above 1.0. The formal `GOAL.md` Gate 3 Docker
protocol remains **closed** on this host; the Docker-gated promotion claim
is deferred to a maintainer or CI rerun with `--engine docker`.

Runtime-budget note frozen before candidate timing: the reference runs
~170-190 s per cell here, safely below the 300-second cap; a timeout would
invalidate the affected pair per protocol.

## Open evidence gaps

- No Docker daemon on the campaign host (Gate 3 closed here).
- No matched external implementation for this exact workload.
- Process peak RSS and XLA memory-traffic estimates not collected in the
  pinned environment.
