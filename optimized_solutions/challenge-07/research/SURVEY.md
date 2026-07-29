# ORBIT-Q Task 07 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-07`

Survey freeze: `2026-07-29T00:04:56Z`

Reference commit: `5af98f27b9404c513df8eee0f4568b1512edee19`

This campaign covers only Task 07. The survey, immutable expert, public
workloads, hypotheses, and measurement rule are frozen before the first
candidate edit.

## Evidence and claim boundary

The immutable human expert is `references/task-07/solution_7.py`
(`sha256:ac483319363f3c386a7646eaa867670ae3d3cd687f8517e6d4201e69240ff0a3`).
The public contract and evaluator are `tasks/task-07/problem.md`
(`sha256:e267f59cdda3d7602ecfdde1a45cb3981e39d52a4bae2f87b4dbb375bcab9680`)
and `tasks/task-07/evaluator/evaluate_7.py`
(`sha256:69717d98a90a7e53c31686128b3ef3e7cea3c96685ec538662a12163fe324b31`).

The historical ORBIT-Q table records 63.8 seconds for Task 07 without a
matched host or environment, and the repository's 2026-07-27 shared-container
bootstrap measured two byte-identical reference cells at
109.332 +/- 0.081 seconds on an eight-CPU/9-GiB allocation. Those values are
context only. No external publication reports this exact evaluator, circuit,
trajectory batch, seed, optimizer trajectory, hardware, and software stack.
This campaign may therefore claim only a paired gain over the bundled expert,
not a global SOTA result.

The problem text contains an internal typo: its displayed objective divides
128 trajectories by 128, while the fixed configuration, interface, evaluator,
and expert all use 64 trajectories and `K.mean`. The executable public
contract is unambiguous. Every candidate must preserve the expert/evaluator's
64-trajectory mean and must not exploit the prose inconsistency.

## Inspected environment and framework paths

Measurements use Docker image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
with six CPUs, 7 GiB memory, no network, and fresh evaluator processes. The
tracked dependency lock is `envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
Per maintainer direction, this campaign uses the image's latest installed
TensorCircuit-NG and does not downgrade it.

| Component | Version | Inspected path / symbol |
| --- | --- | --- |
| TensorCircuit-NG | `1.8.0.dev20260726` | `Circuit.cond_measure`, `conditional_gate`, `state`, and `wavefunction` in `tensorcircuit/circuit.py` (`sha256:5c4d569325369d957dc60bbeca8a581508549ff9813a7a163de61a6294864662`) and `tensorcircuit/basecircuit.py` (`sha256:2f47be7f215c73bfbc41788661b0dd86a77054503f63e7bfbb6c4d2a84004e98`) |
| TensorCircuit measurement templates | same package | `operator_expectation` / `sparse_expectation` in `tensorcircuit/templates/measurements.py` (`sha256:7a69043bd81745254ab106f3cdd0911720fdb91e0be70b784f86c96fbcfaa615`) |
| TensorCircuit sparse operators | same package | `PauliStringSum2COO` in `tensorcircuit/quantum.py` (`sha256:fcaee21ba5ccde1b89c46e2f5424c48d342e3bfaedf72adba672ab6bd4ded703`) |
| JAX / JAXLIB | `0.10.0` / `0.10.0` | `jit`, `vmap`, reverse-mode AD, and `lax.scan`, wrapped by `tensorcircuit/backends/jax_backend.py` (`sha256:88657aebf8e5d566ac4e653abe327083da0253f02a3a297a134b871ffe4baab9`) |
| Optax | `0.2.8` | `optax.adam(0.02)` and `apply_updates` |
| OMECo | `0.2.0` | `TreeSA` contractor shortcuts; the expert requests `omeco-32-32` |
| TensorNetwork-NG | `0.5.1` | TensorCircuit node graph and contraction execution |
| Quimb | `1.11.1` | OMECo path-search support |

TensorCircuit's [measurement documentation](https://tensorcircuit.readthedocs.io/en/stable/faq.html)
states that `cond_measure` performs a normalized Z-basis collapse and returns
a jittable integer tensor, while `conditional_gate` applies a gate selected by
that outcome. Its [Pauli-sum tutorial](https://tensorcircuit.readthedocs.io/en/latest/whitepaper/6-2-pauli-string-expectation.html)
documents both repeated `expectation_ps` evaluation and a single sparse
Hamiltonian `operator_expectation`. The
[TensorCircuit paper](https://quantum-journal.org/papers/q-2023-02-02-912/)
describes the framework's tensor-network, AD, JIT, and vectorization model.
JAX documents `scan` as a single lowered loop for a fixed iteration count in
the [official API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html),
and its [benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)
requires synchronization or host conversion when measuring asynchronous
work. OMECo documents `TreeSA` as a simulated-annealing contraction-order
optimizer in its [public API](https://docs.rs/omeco/latest/omeco/).

## Task 07: measurement-feedback TFIM VQE

### Required algorithm and output

The expert optimizes a 16-qubit, two-layer adaptive circuit. Eight data qubits
and eight ancillas receive trainable `RY` rotations, pairwise `RZZ`
entanglers, and fixed CNOT ladders. Each layer measures all eight ancillas
with fixed per-trajectory uniforms, applies one of two trainable feedback
`RZZ` gates per pair, then applies a data CNOT ladder and trainable `RZ`
rotations. Sixty-four fixed measurement trajectories are mapped with
`K.vmap`, averaged, differentiated, and optimized through exactly 100
sequential Adam updates.

Each trajectory returns the expectation of the open-boundary eight-site TFIM

`H = -sum_i Z_i Z_(i+1) - 1.05 sum_i X_i`.

The solution must return a NumPy `energy_history` of shape `(100,)` containing
pre-update trajectory means and `final_trajectory_energies` of shape `(64,)`
for the same fixed uniforms after the final update.

### Dominant work and measured bottleneck

The expert evaluates every trajectory energy as 15 separate
`Circuit.expectation_ps` calls: seven `ZZ` terms and eight `X` terms. Each
call constructs and contracts a bra/ket tensor network, so the same final
adaptive circuit is effectively contracted 15 times in both the forward and
reverse passes. The final post-training trajectory evaluation repeats the
same pattern.

The immutable expert passes the canonical evaluator in 135.815605 seconds in
the fixed six-CPU container. One update takes 47.853128 seconds end to end,
while 10, 20, 32, and 50 updates take 51.638506, 60.074862, 69.596274, and
91.540316 seconds. Thus roughly 48 seconds is trace/compile/path/finalization
cost, and the remaining 99 canonical updates add about 0.89 seconds each.
Removing Python dispatch alone cannot produce a large gain; the repeated
energy contractions and their gradients are the primary target.

`conditional_gate` also materializes a differentiable dense two-qubit tensor
by one-hot selecting from two `RZZ` gate tensors. After a Z measurement the
ancilla is a computational-basis eigenstate, so this generic representation
retains a two-qubit node even though the gate's action on the data is a
one-qubit phase rotation.

Contraction-path search is inside the timed first trace. The expert chooses
OMECo `TreeSA(ntrials=32,niters=32)`. Contraction-order research establishes
that path choice can substantially change tensor-network work, while better
search also costs more; contractor budget therefore needs end-to-end
measurement rather than FLOP estimates alone. See Schindler and Jermyn,
[Algorithms for Tensor Network Contraction Ordering](https://arxiv.org/abs/2001.08063).

### Exact structural opportunities

At the end of a trajectory, every ancilla was just Z-measured and is touched
only by a diagonal feedback `RZZ`; therefore the final state factorizes as
`|psi_data> tensor |measured_ancilla_bitstring>`. A single TensorCircuit
`c.state()` contraction can be reshaped into `(2^8, 2^8)` and reduced along
the ancilla basis axis to obtain the normalized data state. Feeding that
state to an eight-qubit `tc.Circuit` and TensorCircuit's native sparse
`operator_expectation` evaluates all 15 TFIM terms after one circuit-state
contraction.

For a measured bit `b`, the feedback identity is exact:

`RZZ(theta_b) (|b> tensor |psi>) =
 |b> tensor RZ((1 - 2 b) theta_b) |psi>`.

Replacing the generic conditional two-qubit feedback with the corresponding
TensorCircuit `RZ` on the data qubit preserves the branch, measurement
probability, selected trainable angle, and gradient. It also remains valid
between layers because the feedback never changes the measured ancilla.

### Candidate hypotheses frozen before editing

Every candidate must preserve the seeded float32 initialization, all 96
trainable parameters and their layout, 16 fixed measurement uniforms per
trajectory, normalized `cond_measure` semantics, 64 trajectories and their
order, two layers, exactly 100 Adam updates, pre-update history, final
post-update trajectory values, complex64 behavior, and TensorCircuit as the
central quantum computation.

1. **e01—single native Hamiltonian evaluation.** Contract each final
   trajectory once with `Circuit.state`, extract the factorized data state,
   and evaluate a TensorCircuit-native sparse TFIM operator with
   `templates.measurements.operator_expectation`. Expected value: high.
   Validate energy, parameter gradient, one Adam update, and full history.
2. **e02—measured-ancilla feedback reduction.** Replace each selected
   two-qubit `RZZ` after Z measurement with the exact selected/sign-adjusted
   data `RZ`. Expected value: medium. Validate both branches and full
   trajectory behavior independently before combining it with e01.
3. **e03—whole-training `K.jaxy_scan`.** Carry parameters and Optax state
   through 100 updates and emit the same pre-update values. Expected value:
   small to medium because it removes host dispatch but not quantum work.
4. **e04—contractor budget.** Compare the frozen best circuit under
   OMECo 1x1, 4x4, 8x8, 16x16, and 32x32 or greedy where supported.
   Path-search time and all repeated contractions must be measured together.
5. **e05—further state/measurement reuse.** Contract once before an
   eight-ancilla measurement round and derive sequential conditional
   probabilities from the TensorCircuit state, then reinitialize a
   TensorCircuit circuit from the collapsed branch. Potential value: high,
   but risk is also high because fixed-uniform sequential collapse and
   framework-fidelity boundaries must remain exact. Pursue only after the
   lower-risk native changes.

The ideas are separate falsifiable hypotheses. No candidate implementation
was edited before this survey and dataset freeze.

## Frozen measurement and promotion rule

All eligible comparisons use one long-lived container with six CPUs and
7 GiB, the image ID above, no network, a 300-second per-cell cap, and a fresh
evaluator process per cell. Six matched pairs will be run, exceeding the
user's five-run requirement:

- odd pairs: reference then candidate;
- even pairs: candidate then reference.

Report every runtime, arithmetic mean, median, sample standard deviation,
standard error, minimum, maximum, ratio-of-means improvement, and each
pairwise speedup `S_i = R_i / C_i`. The primary confidence interval is the
two-sided 95% Student-t interval on the arithmetic mean of pairwise speedups:

`mean(S) +/- t_(0.975,5) * sample_stdev(S) / sqrt(6)`,

where `t_(0.975,5)=2.5705818366`.

Promotion requires all 12 cells to pass, candidate mean and median below the
reference, at least five of six pair wins, and a confidence-interval lower
bound above 1.0. The canonical 100-step workload is the claim workload; the
public 50-step passing workload is only for screening and robustness.

## Open evidence gaps

- No matched external implementation/hardware runtime exists for this exact
  adaptive VQE evaluator.
- Peak intermediate memory and contractor-estimated FLOPs are not yet
  recorded.
- The current full canonical expert result is one bootstrap run; six
  counterbalanced reference cells will be collected only against the frozen
  winning candidate.
- Results will apply only to the fixed eight-data/eight-ancilla,
  two-layer/64-trajectory workload on this host; no scaling claim is planned.
