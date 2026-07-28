# ORBIT-Q Task 05 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-05`

Survey freeze: `2026-07-27T18:15:35Z`

Reference commit: `46d6636881500fa8f70618b74f89353a2b6702b4`

This survey covers only Task 05, as required by the one-task campaign scope in
`GOAL.md`. `READY` means the public knowledge and measurement plan are
complete. It does not attest that the public dataset, trusted controller, or
repeated-runtime gates have passed.

## Evidence and claim rules

Repository evidence below names a path, symbol or line, commit, and SHA-256.
Pinned-environment evidence was inspected in the network-disabled benchmark
image `sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c`.
External evidence links primary project documentation or papers.

The bundled immutable human expert is the only reproducible performance
comparator currently available for this exact 18-qubit, ten-layer, 600-update
workload. The historical ORBIT-Q values are context only. No cited external
paper reports a matched evaluator runtime, hardware allocation, software
version, and output contract. Therefore this campaign may claim a paired gain
or a 10x gain *over the bundled human expert* when the frozen rule passes, but
must not claim global state of the art without a new matched external
comparator.

## Pinned environment and inspected framework paths

The package versions come from
`envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
The Dockerfile is
`envs/tensorcircuit-py311/Dockerfile`
(`sha256:cbcffdfc0f17731bcdf549b94209879043aab8e53ee937be60947a49d2065912`).

| Component | Pinned version | Inspected source or API |
| --- | --- | --- |
| TensorCircuit-NG | `tensorcircuit-nightly==1.7.0.dev20260618` | `/usr/local/lib/python3.11/site-packages/tensorcircuit/circuit.py:754`, `Circuit.wavefunction/state` (`sha256:45151c19d606c66cf209e903a3bcf927f2e266ea9d4facfb9332a16dec7ec8ef`); `tensorcircuit/cons.py:1136`, `set_contractor` (`sha256:4fb517ad6085328c53ed61c75471579708eb8265fa6c14ec98e4bf1b4fd04365`); `tensorcircuit/quantum.py:1919`, `PauliStringSum2MVP` (`sha256:550e2cc7d41b4eadfe7ce6dd37f1d295539eb22a580f5e2993a0a2ec3408c1b4`) |
| JAX | `0.10.0` | TensorCircuit wrapper `/usr/local/lib/python3.11/site-packages/tensorcircuit/backends/jax_backend.py:712,803,844`, `scan`, `value_and_grad`, and `jit` (`sha256:22bbcf80d5a018884337286f176849ad4e8681b4e1f20efc1f80ba9ec4cc9d69`) |
| JAXLIB | `0.10.0` | Version confirmed from the pinned image; CPU device selected by the benchmark image and host allocation |
| OMECo | `0.2.4` | Registered through TensorCircuit `set_contractor`; tracked Task 05 variant calls `tc.set_contractor("omeco")` |
| TensorNetwork-NG | `0.5.1` | TensorCircuit `Circuit.state` copies nodes/front edges and calls the configured TensorNetwork contractor |
| Quimb | `1.11.1` | Installed support library; the immutable Task 05 expert does not call it |
| Optax | `0.2.8` | `/usr/local/lib/python3.11/site-packages/optax/_src/alias.py:414`, `adam`; the public [Optax Adam API](https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html) defines the first/second-moment state used by the expert |
| Benchmark image | `orbitbreakers-expert-benchmarks:tensorcircuit-py311` | Image ID `sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c`; 8 CPU and 9 GiB limits from `bench.toml` |

TensorCircuit documents JAX, automatic differentiation, JIT, vectorization,
and contractor selection as supported execution paths in its
[quick start](https://tensorcircuit.readthedocs.io/en/latest/quickstart.html).
Its contractor documentation says the framework is a tensor-network
contraction simulator and describes preprocessing that merges one-qubit gates
into neighboring entangling gates. The
[contractor tutorial](https://tensorcircuit.readthedocs.io/en/latest/tutorials/contractors.html)
identifies contractor choice as a space-time tradeoff and shows reusable
Cotengra optimizers.

## Task 05: custom non-unitary gate cooling

### Immutable expert and output contract

The immutable expert is
`references/task-05/solution_5.py`
(`sha256:ccafe626865ee39b651adaeead86b8bf6f541e3f1426da4842da92b6a0ee015f`).
The contract and timer are
`tasks/task-05/problem.md`
(`sha256:b0b31d059b86d92bee03a51a82586c0c4e0823de2ec5a96dc5772baeccd5da78`)
and `tasks/task-05/evaluator/evaluate_5.py:evaluate`
(`sha256:dd0742cf402827beec19328bc9cf090e80a08973cf9303fd7d524a4f4cd37402`).

The evaluator starts `time.perf_counter()` immediately before
`run_solution(config)` and stops after it returns. The timed region therefore
includes tracing/compilation triggered by the call, construction of the
initial state and TFIM matrix-vector product, all 600 gradient/update steps,
device synchronization caused by NumPy conversion, and result assembly. The
independent sparse eigensolve and functional checks occur after the timed
region.

The expert:

1. prepares `|+>^18` with TensorCircuit;
2. represents ten shared-parameter cooling layers as five even/odd blocks;
3. applies `RX(2 i a_l)` on all 18 qubits and `RZZ(2 i b_l)` on 9 even or 8
   odd brickwork bonds;
4. contracts a full state and divides by its norm after every layer;
5. evaluates the open-boundary TFIM energy density using
   `PauliStringSum2MVP`;
6. differentiates through all contractions and normalizations; and
7. performs exactly 600 `optax.adam(0.02)` updates while recording the
   pre-update energy at every step.

It must return NumPy-compatible `final_a` and `final_b` arrays of shape
`(5, 2)` plus a length-600 `energy_density_history`. Every candidate must
preserve these values' meaning, the layer/bond order, complex64
TensorCircuit-NG semantics, ten normalizations with differentiation, and the
exact Adam trajectory definition.

The physical method is related to non-unitary imaginary-time filtering. Xie
et al., [“A Probabilistic Imaginary Time Evolution Algorithm Based on
Non-unitary Quantum Circuit”](https://arxiv.org/abs/2210.05293), apply
non-unitary circuits to ground-state preparation including an Ising chain.
Leadbeater et al.,
[“Non-unitary Trotter circuits for imaginary time evolution”](https://arxiv.org/abs/2304.07917),
also study non-unitary imaginary-time primitives on the transverse Ising
model. These papers support the algorithmic family; neither is a matched
runtime comparator for this evaluator.

### Scaling, memory, and dominant work

For `n=18`, the dense complex64 wavefunction has `2^18 = 262,144` amplitudes
and occupies 2 MiB before autodiff intermediates. One forward trajectory
contains 180 one-qubit gates, 85 two-qubit gates, and ten full-state norm
reductions. The TFIM has 17 `ZZ` and 18 `X` terms. The inspected
`PauliStringSum2MVP` implementation loops over those 35 terms, forming each by
broadcasted phase multiplication and/or axis-reversing slices, then sums the
full-state results.

For a dense exact state, a local gate, norm, or Pauli-term application is
`O(2^n)` in time. The source-level forward bound is therefore
`O((L n + T) 2^n)` for `L=10` layers and `T=35` Hamiltonian terms, with a
constant-factor reverse-mode pass for the scalar loss. The live-memory lower
bound is `O(2^n)`; reverse-mode differentiation may retain or recompute
layer/gate intermediates depending on JAX/XLA lowering.

Local bootstrap evidence in `baselines/bootstrap-2026-07-27.md` reports two
byte-identical Task 05 pairs at `116.277 ± 3.728 s` for the reference and
`115.078 ± 2.944 s` for the initial optimized copy under the same 8-CPU/9-GiB
image. Those two pairs measure parity/noise only and are ineligible for a
speedup claim. `baselines/historical.json` records an informational publication
runtime of 45.5 s and an unmatched local OMECo variant at 34.85 s versus a
48.27 s expert record. Both records are hypothesis evidence, not current SOTA
or promotion evidence.

The immutable-reference profile
`research/profiles/task-05-reference-profile.json`
(`sha256:be24858b7693ff10c1c153a7fb27ba73a2b60fa7eae5e74ea16be9aa74e6473c`)
measured an eight-step steady mean of `0.169716 s`, projecting to `101.830 s`
for 600 executions versus the six-run evaluator mean of `117.776 s`. The
compiled update's XLA analysis reports about 210.0 million FLOPs, 1.187 GB of
memory traffic, and 289.5 MB of temporary storage per step. Lowering and
compilation took `0.590 s` and `1.287 s`; the first compile-plus-execute call
took `1.982 s`. Thus steady gradient/update execution accounts for roughly
86.5% of the baseline, and compilation or Python dispatch alone cannot produce
a 10x result.

The profile also exposed a framework hazard: `PauliStringSum2MVP` keeps a
mutable dtype cache. Reusing one MVP closure across independent JAX traces
caused a tracer leak. A candidate must use one stable transformed loss or a
fresh MVP closure per independent trace.

The principal source- and profile-supported bottlenecks are:

- 600 scalar-loss reverse-mode evaluations over dense `2^18` states;
- repeated tensor-network construction/contraction for each normalized layer;
- a 35-term full-state Hamiltonian MVP;
- approximately 1.187 GB of XLA-estimated memory traffic per gradient/update;
- full-state reverse-mode storage/recomputation.

JAX recommends placing `jax.jit` on the outermost useful function in its
[benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html).
The expert JIT-compiles one update but dispatches it from a Python loop 600
times. JAX documents that
[`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
lowers to a single While operation and avoids Python-loop unrolling inside a
JIT. The final `K.numpy` conversions synchronize the asynchronous work, so the
evaluator's end-to-end timer includes execution; JAX's
[asynchronous dispatch documentation](https://docs.jax.dev/en/latest/async_dispatch.html)
explains this host-read synchronization behavior.

### Twenty predeclared optimization hypotheses

Each round changes only
`src/solutions/task-05/solution_5.py`, starts from the latest accepted commit,
and must preserve the semantics above. A hypothesis that requires prohibited
raw-JAX simulation, fewer optimizer updates, omitted normalization, altered
precision, or hidden-data access is rejected before implementation and remains
recorded as a failed/rejected round.

1. **Whole-training scan:** JIT one `lax.scan` over all 600 Adam updates to
   reduce Python dispatch while returning every pre-update energy. The
   reference profile makes this a diagnostic/secondary hypothesis rather than
   a plausible standalone 10x route.
2. **TensorCircuit backend scan:** express the same loop with
   `K.jaxy_scan`/`K.scan` to test whether the framework wrapper lowers more
   cleanly than direct JAX.
3. **OMECo contractor:** use the pinned `omeco==0.2.4` contractor, motivated by
   the unmatched tracked precedent, without changing circuit operations.
4. **Greedy preprocessing:** configure TensorCircuit contraction preprocessing
   to merge one-qubit gates into neighboring entanglers.
5. **Plain state-simulator contractor:** test `plain-experimental`, which the
   TensorCircuit quick start identifies as a state-simulator-like contractor.
6. **Reusable Cotengra path:** preconfigure a deterministic reusable greedy
   path optimizer with a bounded search time included in `run_solution`.
7. **Contractor ablation on whole-training scan:** combine the best legal
   contractor with round 1 and test whether their gains compose.
8. **Array parameter layout:** replace the two-leaf `{a,b}` PyTree with one
   `(2,5,2)` TensorCircuit-backend array to reduce optimizer/tree overhead while
   keeping element order and Adam states identical.
9. **Tuple scan carry:** remove dictionaries from the compiled carry and use a
   fixed tuple `(params,opt_state)` to reduce trace/tree overhead.
10. **Static bond tuples:** close over immutable even/odd bond tuples and all
    static configuration scalars so JAX sees no per-step Python construction.
11. **Single initial-state tensor construction:** create `|+>^18` once with
    TensorCircuit and close it over the compiled loss, as the expert intends,
    while checking for accidental retracing.
12. **Hamiltonian construction ablation:** compare the current
    `PauliStringSum2MVP` closure with TensorCircuit's supported Pauli
    expectation API, preserving the exact 35 terms and scalar energy.
13. **Sparse TensorCircuit Hamiltonian:** test TensorCircuit's pinned sparse
    COO path only if it remains differentiable and avoids dense `2^n × 2^n`
    materialization.
14. **Layer rematerialization:** apply JAX checkpoint/rematerialization at one
    normalized layer to trade recomputation for a lower memory working set.
15. **Block rematerialization:** checkpoint one even+odd block rather than each
    layer and compare the XLA memory/runtime tradeoff.
16. **Exact MPS circuit:** test TensorCircuit `MPSCircuit` with no truncation;
    reject if it changes gate, normalization, gradient, or complex64 semantics.
17. **Exact split-gate MPS:** if round 16 is valid, use TensorCircuit's
    documented split-two-qubit-gate path with zero truncation error.
18. **Compiled value/gradient boundary:** construct one stable
    `K.value_and_grad(loss_fn)` object before JIT so no function object is
    recreated during the scan.
19. **Donation-safe carry:** test legal JAX buffer donation only if
    TensorCircuit's wrapper and returned history remain correct; otherwise
    record the aliasing incompatibility.
20. **Best-candidate consolidation:** combine only independently valid,
    composable wins, then run the full six-pair promotion and public functional
    checks as an ablation-backed final round.

OMECo, TensorNetwork-NG, and Quimb remain support paths. They may optimize
TensorCircuit-NG contraction but may not replace the central TensorCircuit
calculation. A free-fermion, hand-written statevector, hard-coded trajectory,
precomputed answer, reduced step count, or looser evaluator is outside the
campaign.

## Frozen paired measurement and statistics plan

This section is frozen before candidate timing results.

- Workload: public dataset version recorded in
  `datasets/public/manifest.json`; Task 05 canonical fixed configuration only.
- Engine/allocation: pinned Docker image above, one 8-CPU/9-GiB container per
  experiment, network disabled.
- Cells: six matched pairs minimum; a fresh evaluator process per cell in the
  same task container.
- Order: odd pairs `reference → candidate`; even pairs
  `candidate → reference`.
- Timeout: hard 300 seconds per evaluator process. A timeout stops the shared
  container and invalidates the incomplete comparison.
- Cache state: cold process-local Python/JAX caches for every fresh process;
  no candidate-only precomputation outside `run_solution`. Any persistent
  compilation-cache experiment must be run as a separately declared matrix
  with identical cache policy for both roles and cannot be pooled with cold
  runs.
- Validity: process exit zero, finite positive evaluator runtime, `Overall:
  PASS`, full output schema, exact 600-step history, and framework-fidelity
  audit.

For eligible runtimes `R_i` and `C_i`, report means, medians, sample standard
deviations, standard errors, minima, maxima, ratio-of-means improvement
`100*(mean(R)-mean(C))/mean(R)`, pairwise improvements
`100*(R_i-C_i)/R_i`, and pairwise speedups `S_i=R_i/C_i`.

The predeclared confidence interval is the two-sided 95% Student-t interval for
the arithmetic mean of the pairwise speedups:

`mean(S) ± t_(0.975,n-1) * sample_stdev(S)/sqrt(n)`.

With six pairs, the critical value is `t_(0.975,5)=2.5705818366`. Report the
pairwise-improvement and speedup standard errors as sample standard deviation
divided by `sqrt(n)`. Do not change the method after observing a candidate.

A candidate is promotable only when every validity/fidelity gate passes, at
least six pairs are eligible, candidate mean and median are lower, the
candidate wins at least 80% of pairs, and the 95% lower confidence bound for
mean paired speedup exceeds 1.0. A 10x result additionally requires the
recorded paired evidence to support mean speedup at least 10 and its 95% lower
bound to exceed 10. No failed, timed-out, unmatched, historical, or hidden
per-record result contributes to a runtime claim.

## Open evidence gaps

- No matched external implementation publishes runtime for this exact Task 05
  configuration, output history, TensorCircuit-NG version, CPU allocation, and
  evaluator timer. Global SOTA remains unestablished.
- Process peak RSS and per-kernel CPU profile are not yet measured. XLA cost,
  temporary-buffer size, and compilation share are measured by the public
  immutable-reference profile.
- OMECo's historical local gain is unmatched and predates this frozen paired
  protocol.
- The trusted controller must independently attest hidden tuning rotations and
  a sealed holdout for the public dataset version before round 1 can change
  candidate code.
