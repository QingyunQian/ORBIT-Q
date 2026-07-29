# ORBIT-Q Task 04 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-04`

Survey freeze: `2026-07-29T07:56:11Z`

Reference commit: `5af98f2e`

This survey covers only Task 04. `READY` means that the public knowledge,
framework inspection, hypotheses, and measurement rule are frozen before
candidate timing. It does not itself promote a candidate.

## Evidence and claim boundary

The immutable human expert is the only reproducible performance comparator for
the exact fixed Task 04 contract. No external publication reports an
evaluator-compatible runtime for the same four probes, asymmetric channel,
observable table, 120-step Adam trajectory, hardware, and software image.
Accordingly, this campaign may claim a paired improvement over the bundled
expert, but not global state of the art.

General algorithmic context comes from primary or official sources:

- TensorCircuit is a tensor-network quantum-circuit framework designed around
  contraction, differentiability, JIT, and vectorization
  ([TensorCircuit paper](https://arxiv.org/abs/2205.10091)).
- Exact circuit contraction cost is governed by contraction structure and
  treewidth rather than only the nominal Hilbert-space dimension
  ([Markov and Shi](https://arxiv.org/abs/quant-ph/0511069)).
- TensorCircuit documents full `DMCircuit` simulation as a doubled-index
  representation and documents manual Kraus operators through
  `general_kraus`
  ([density-matrix guide](https://tensorcircuit.readthedocs.io/en/stable/whitepaper/5-density-matrix.html)).
- TensorCircuit's official quick start recommends the JAX backend for AD, JIT,
  and VMAP, and explicitly supports vectorizing circuit inputs, parameters, and
  measurements
  ([quick start](https://tensorcircuit.readthedocs.io/en/stable/quickstart.html),
  [VMAP guide](https://tensorcircuit.readthedocs.io/en/latest/whitepaper/6-3-vmap.html)).
- JAX recommends JIT at the outermost useful boundary and host synchronization
  for trustworthy timing
  ([benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html)).
  `lax.scan`, exposed in this image through TensorCircuit `K.jaxy_scan`, lowers
  a fixed loop to a single WhileOp
  ([JAX scan API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)).

## Pinned measured environment and inspected framework

The benchmark ran in image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
with 6 CPUs, 7 GiB memory, no network, and a 300-second per-cell timeout.
The tracked lock is
`envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`);
the existing derived image intentionally contains the newer
TensorCircuit-NG build requested for this campaign.

| Component | Measured version | Inspected source/API |
| --- | --- | --- |
| TensorCircuit-NG | `1.8.0.dev20260726` | `/usr/local/lib/python3.11/site-packages/tensorcircuit/densitymatrix.py`: `DMCircuit.apply_general_kraus`, `DMCircuit2.apply_general_kraus`, `densitymatrix`, `expectation`; `basecircuit.py`: `_copy_state_tensor`, `expectation_before`; `mpscircuit.py`: `apply_MPO`, `expectation` |
| JAX / JAXLIB | `0.10.0` / `0.10.0` | `tensorcircuit/backends/jax_backend.py`: `jit`, `value_and_grad`, `vmap`, `jaxy_scan` |
| TensorNetwork-NG | `0.5.1` | Node/copy/contractor substrate used by the TensorCircuit circuit classes |
| Optax | `0.2.8` | `optax.adam(0.04)` and `optax.apply_updates` |
| OMECo | `0.2.0` in the measured image | Optional TensorCircuit contractor path; not assumed beneficial without a paired ablation |
| Quimb | `1.11.1` | Installed support library; not used by the immutable Task 04 expert |

The inspected current source exposes an important distinction. Legacy
`DMCircuit.apply_general_kraus` first calls `_contract`, copies the entire
density-matrix circuit once per Kraus branch, materializes every branch
density matrix, and sums them. `DMCircuit2.apply_general_kraus` instead turns
the Kraus list into a local superoperator node and leaves it in the tensor
network for later contraction. This is the strongest TensorCircuit-native
hypothesis because it changes the contraction schedule, not the channel.

## Task 04: trainable Kraus noise calibration

### Immutable expert and exact contract

The immutable source is `references/task-04/solution_4.py`
(`sha256:04e37b73e7246599ed3eb8f65e38bb7e084db7aab511d7af3b20baa0867b21ae`).
The contract is `tasks/task-04/problem.md`
(`sha256:8724b467812e5fca2be6f8c6b6157e9e083fd1b318149577d54fe4337f41f4ea`)
and the timer/checks are `tasks/task-04/evaluator/evaluate_4.py`
(`sha256:7f7e215064695ba3d0a0f5fb4883d78f7d225b2b7a79db70cdf3a0f3228314c5`).

The timed function must:

1. generate the target table inside `run_solution` at true
   `(p01, p10) = (0.034, 0.011)`;
2. prepare GHZ, Bell-pair, zero, and plus probes on 12 qubits;
3. apply 6 even-bond and 5 odd-bond `RXX(0.31)` gates, followed after every
   entangler by the user-defined three-Kraus asymmetric bit-flip channel on
   each participating qubit;
4. return all 12 single-site Z expectations and full Z parity for each probe;
5. optimize sigmoid-parameterized probabilities from `(0.070, 0.040)` for
   exactly 120 Adam updates at learning rate `0.04`;
6. record the pre-update loss for every update; and
7. return NumPy `loss_history`, `final_probabilities`, and
   `fitted_expectations` with shapes `(120,)`, `(2,)`, and `(4,13)`.

Every candidate must preserve the channel's Kraus tensor algebra, fixed probe
states, gate/channel order, all observables, complex64 TensorCircuit semantics,
gradient path, Adam trajectory meaning, and final post-update probabilities.
An analytic closed-form classical fit, hard-coded table, fewer steps, Monte
Carlo replacement, raw NumPy/JAX simulator, or evaluator-specific answer is
outside this campaign.

### Scaling and source-supported bottleneck

A 12-qubit density matrix contains `4^12 = 16,777,216` complex64 entries,
128 MiB before autodiff temporaries. The circuit has 22 one-qubit channel
applications per probe: 12 after the 6 even bonds and 10 after the 5 odd
bonds. One observable table therefore invokes 88 channels. The target plus
120 fitted evaluations expose 10,648 channel applications at source level.

The legacy implementation's `DMCircuit.apply_general_kraus` eagerly contracts
before each of those channels and materializes three full-state branches.
That makes repeated dense `O(4^n)` traffic the primary source-supported cost.
The observable loop performs 52 expectations per table, but after the eager
channels the circuit already consists primarily of one dense density tensor;
observable-only tuning is therefore secondary until measured otherwise.

The complete immutable-reference baseline on this host is:

```text
15.250249, 14.381873, 14.605714, 14.544423, 14.697870, 14.546563 s
mean 14.671115 s; median 14.576139 s; stderr 0.123236 s; 6/6 PASS
```

It is stored at
`results/task-04-reference-20260729-v2/results.json`
(`sha256:be8696214fa5c170398cd7a57bb3df56a476e658ebca5c99f8ec7a487735ff16`).
The earlier bootstrap's two byte-identical runs are timing-noise context only.

The immutable-reference profile
`research/task-04/profiles/reference-profile.json`
(`sha256:eecfa9259600492fe7a8b3b3e9a4fa17a55df13395a41142683f9ece16001ebf`)
separates the dominant stages. Target-table generation took `2.629 s`; lowering
one expert update took `3.068 s`; XLA compilation took `10.712 s`; eight
compiled steady updates averaged `1.257 ms`, projecting to only `0.151 s` for
120 executions. The lowered update has 21,720 StableHLO lines. Thus the
canonical end-to-end runtime is compilation dominated. A whole-training scan
cannot be the leading factor here; reducing the traced noisy-circuit graph and
target construction has much higher expected value.

## Predeclared optimization hypotheses

Each factor is evaluated separately before consolidation.

1. **TensorCircuit `DMCircuit2`:** replace eager three-branch density
   materialization with native local Kraus-superoperator TN nodes.
2. **Scalar TN expectations:** use `DMCircuit2` with `reuse=False` so each
   observable is a scalar TN contraction instead of forcing a full
   `4^12` output tensor.
3. **Deferred-state reuse:** compare `reuse=True`, which contracts the noisy
   state once and reuses it across 13 observables, against scalar contraction.
4. **Native contractor choice:** compare TensorCircuit default, greedy,
   OMECo, and a bounded reusable Cotengra optimizer when path-search overhead
   remains inside the timer.
5. **Probe VMAP:** provide four TensorCircuit-compatible initial tensors and
   batch the same noisy circuit through `K.vmap`.
6. **Measurement VMAP:** vectorize the 13 observable insertions if it reduces
   repeated host tracing without materializing a larger dense state.
7. **Whole-training scan:** express all 120 Adam updates with
   TensorCircuit-native `K.jaxy_scan`, preserving pre-update history and final
   post-update parameters.
8. **Jitted target and training boundary:** compile target generation and the
   complete training trajectory at the outermost stable TensorCircuit
   boundary to prevent per-step Python dispatch.
9. **Single two-qubit noisy-layer superoperator:** compose `RXX` plus the two
   independent one-qubit Kraus maps into one exact rank-4 local superoperator
   through TensorCircuit Kraus/super-gate APIs.
10. **Kraus construction simplification:** stack the three channel tensors
    with fewer scalar casts/stacks while keeping the same differentiable
    complex64 matrices.
11. **Static probe tensors:** construct the four fixed input TNs once per
    `run_solution` and feed them to the parameterized noisy network.
12. **MPO density state:** use TensorCircuit `QuOperator`/MPO inputs and exact
    local superoperators, with no truncation, if `DMCircuit2` scalar
    contraction is not sufficient.
13. **Locally purified MPS:** test a TensorCircuit `MPSCircuit` purification
    only if every channel environment and observable can remain exact and
    differentiable.
14. **Heisenberg causal cone:** contract each observable backward through the
    same TensorCircuit gate and Kraus tensors; reject any handwritten
    classical formula that replaces the framework computation.
15. **Parity MPO:** represent full-chain Z parity as a bond-1 TensorCircuit
    MPO to avoid dense operator construction.
16. **Shared contraction path:** reuse a deterministic scalar-network
    contraction path across fitted parameters and gradients.
17. **Two-parameter forward-mode check:** compare a TensorCircuit backend
    forward derivative only if the exact Optax gradient structure is
    preserved.
18. **Tuple optimizer carry:** use a fixed `(params, opt_state)` scan carry
    and one stable transformed loss object.
19. **Donation/rematerialization:** test only if supported through the
    TensorCircuit backend and beneficial under the measured 7 GiB limit.
20. **Final consolidation and leave-one-out ablation:** combine only
    independently valid wins, then remove each retained factor in turn.

The first experiment is `DMCircuit2` alone because source inspection predicts
the largest reduction in full-density materialization while staying entirely
inside TensorCircuit-NG.

## Frozen paired measurement and promotion rule

- Workload: `orbitq-workloads-v20260729.1`,
  `task-04-canonical-fixed-v1`.
- Engine: Docker image and 6 CPU / 7 GiB allocation recorded above.
- Timer: evaluator-reported end-to-end `run_solution` time, including tracing,
  compilation, all 120 updates, synchronization from NumPy conversion, and
  result assembly.
- Cells: at least six matched alternating pairs in one task container; odd
  pairs reference then candidate, even pairs candidate then reference.
- Process state: every cell is a fresh evaluator process; no candidate-only
  cache or work outside `run_solution`.
- Timeout: 300 seconds per evaluator process.
- Validity: exit zero, finite positive runtime, `Overall: PASS`, exact output
  shapes/history length, probability tolerance, trace-preserving check, and
  TensorCircuit-NG framework fidelity.

For pair `i`, speedup is `S_i = R_i / C_i`. Report means, medians, sample
standard deviations, standard errors, minima, maxima, ratio-of-means
improvement, every paired speedup, and the two-sided 95% Student-t interval

`mean(S) ± t_(0.975,n-1) * sample_stdev(S) / sqrt(n)`.

For six pairs, `t_(0.975,5)=2.5705818366`. Promotion requires six eligible
pairs, lower candidate mean and median, at least 80% pair wins, and a 95%
lower confidence bound above 1. The first-five average is also reported for
the user's five-run request; the complete six-pair result remains
authoritative.

## Open evidence gaps

- No matched external implementation exists for this exact evaluator, so all
  performance claims are relative to the bundled expert.
- Generic TN contraction complexity can be much better than a dense density
  matrix for shallow circuits, but the four probe preparations and parity
  observable give different contraction graphs; actual contractor behavior
  must be measured.
- The installed `DMCircuit2` API is less documented than `DMCircuit`; reduced
  state, observable, loss, gradient, one-update, and full-history equivalence
  are mandatory before promotion.
- Only the fixed canonical 12-qubit workload is claimable; no scaling claim is
  made without additional public workload support.

## Post-freeze measured-image binding correction

The predeclared first hypothesis above was based on the distinct legacy class
bodies in `densitymatrix.py`. Runtime introspection after e01 established that
the measured `1.8.0.dev20260726` package exports both public names as the same
class:

```text
tc.DMCircuit  = tensorcircuit.densitymatrix.DMCircuit2
tc.DMCircuit2 = tensorcircuit.densitymatrix.DMCircuit2
```

Thus the immutable expert already uses local Kraus-superoperator TN nodes in
this image. The e01 rename is a no-op, and all final bottleneck/strategy claims
are based on the measured 21,720-line traced graph, probe VMAP, and explicit
local-node consolidation rather than eager legacy density materialization.
