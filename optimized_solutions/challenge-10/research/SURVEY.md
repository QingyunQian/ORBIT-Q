# ORBIT-Q Task 10 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-10`

Survey freeze: `2026-07-29`

Reference repository commit: `7be259106c9dae899e5ea8f82c2251ca285640a4`

This survey covers only Task 10. The immutable comparator is the published
human-expert solution in `references/task-10/solution_10.py`; the editable
candidate remains byte-identical while the survey and public-workload gates
are prepared.

## Evidence and claim boundary

The exact 22-qubit, two-layer, 200-update workload has no cited matched
external runtime result with the same evaluator, dependency lock, and CPU
allocation. The campaign may therefore claim an improvement over the bundled
human expert after the frozen paired protocol passes, but it must describe the
winner as the **campaign-best implementation**, not a global SOTA.

The Task 10 reference SHA-256 is
`0e3266857e4faa8a4d65092b0e88c2866042d716cb0ef8a278633a4f30bb6172`.
The evaluator SHA-256 is
`0ab012597cfa79ec32ebc55bb28307c7b15309315f8057604760df5ad4be71db`.
The public problem SHA-256 is
`a5e96735db39f9b19c314c31fe1b6bc63a6b8c90ff5cd02b0d8f9a6b4df65584`.

The ORBIT-Q open-PR search was inspected on 2026-07-29 before the campaign.
PRs 4, 5, and 6 are whole-suite benchmark-result PRs; none proposes a Task 10
human-expert performance improvement. Task 10 is therefore eligible under the
one-task campaign rule.

## Pinned environment and inspected framework paths

The benchmark lock is
`envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
It pins Python 3.11, `tensorcircuit-nightly==1.7.0.dev20260618`, JAX/JAXLIB
0.10.0, Optax 0.2.8, Quimb 1.11.1, OMECo 0.2.4, and TensorNetwork-NG 0.5.1.
The OMECo compatibility shim is
`envs/tensorcircuit-py311/sitecustomize.py`
(`sha256:02800060761f2b15abe9055aded49d2af3877ab93d7b0fae8af94b30bac30120`);
it reproduces TensorCircuit-NG commit
`53a712b517cdcaba69ca6376d9d68cd140bdeaea` by registering an OMECo TreeSA
contractor with 16 trials, 32 iterations, the published score weights, and
TensorCircuit preprocessing.

The current TensorCircuit-NG source was also inspected at commit
`b886a5e3a915fdb660d630216d186b96dca2eccb`:

- `tensorcircuit/gates.py::cmz_gate` represents the multi-controlled Z diagonal
  as a bond-2 chain, avoiding a dense `2^18 x 2^18` gate;
- `tensorcircuit/templates/measurements.py::mpo_expectation` contracts
  `bra @ MPO @ ket` as one tensor network;
- `tensorcircuit/mpscircuit.py::apply_MPO` applies an MPO to an MPS through
  framework backend contractions and QR/SVD canonicalization;
- `tensorcircuit/mpscircuit.py::expectation` evaluates local operators directly
  on an MPS;
- `.agents/skills/performance-optimize/SKILL.md` recommends native
  representations, moving static physics out of the hot path, whole-step JIT,
  scans for repeated work, and separate staging/steady-state measurements.

Primary source links:

- [TensorCircuit-NG repository and framework overview](https://github.com/tensorcircuit/tensorcircuit-ng)
- [CMZ bond-2 implementation](https://github.com/tensorcircuit/tensorcircuit-ng/blob/b886a5e3a915fdb660d630216d186b96dca2eccb/tensorcircuit/gates.py)
- [MPS circuit implementation](https://github.com/tensorcircuit/tensorcircuit-ng/blob/b886a5e3a915fdb660d630216d186b96dca2eccb/tensorcircuit/mpscircuit.py)
- [MPO expectation implementation](https://github.com/tensorcircuit/tensorcircuit-ng/blob/b886a5e3a915fdb660d630216d186b96dca2eccb/tensorcircuit/templates/measurements.py)
- [JAX `lax.scan` API](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)
- [Optax Adam API](https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html)
- [TensorCircuit software paper](https://doi.org/10.22331/q-2022-02-02-657)

## Task 10: 22-qubit VQE with an 18-qubit controlled-Z hyperedge

### Immutable expert algorithm and output contract

The canonical configuration fixes:

- 22 qubits and the listed 18 non-adjacent CMZ sites;
- the alternating computational-basis initial state;
- two `RX -> RZ -> RY` rotation blocks, each followed by one framework
  `cmz` gate;
- the open-boundary TFIM Hamiltonian
  `-zz * sum(Z_i Z_{i+1}) - x * sum(X_i)`;
- NumPy `default_rng(seed)` float32 initialization with shape `(2, 22, 3)`;
- exactly 200 Adam updates at learning rate 0.03;
- pre-update energy history of shape `(200,)` and final parameters of shape
  `(2, 22, 3)`.

Every candidate must preserve the gate order, CMZ semantics, seeded
initialization, optimizer update count, pre-update history convention, output
keys and shapes, finite values, variational lower bound, and evaluator energy
gap. The central quantum computation must remain TensorCircuit-NG.

The human expert builds a Quimb bond-3 TFIM MPO, converts it with
`tc.quantum.quimb2qop`, constructs the full TensorCircuit network for each
energy call, evaluates it with `mpo_expectation`, differentiates the scalar
energy, and JIT-compiles the whole Adam step. A Python loop dispatches that
compiled step 200 times.

### Cost model and supported bottlenecks

The ansatz is unusually favorable to an exact MPS representation. Single-site
rotations do not increase bond dimension. A CMZ is a bond-2 diagonal MPO, so
after two exact applications the state bond dimension is bounded by four
before any rank compression. The TFIM has only 21 nearest-neighbor ZZ terms
and 22 one-site X terms, each contractible in
`O(n * chi^3)` or better with `chi <= 4`.

By contrast, the expert asks a general contraction-path optimizer to plan the
combined circuit-bra/MPO/circuit-ket network during JAX tracing. This is
mathematically valid and already much better than a dense 22-qubit
statevector, but it carries generic network construction, OMECo path search,
and a larger compiled contraction graph.

Historical same-machine evidence in the parent ORBIT-Q run reports
approximately 20.6 seconds for the expert and 372.09 seconds for a dense-state
candidate. A fresh semantic-validation run on 2026-07-29 using the available
`challenge-benchmark-quantum-tensorcircuit:py311` image completed the immutable
expert in 18.117244 seconds and passed every evaluator criterion. These are
context and gate evidence, not the final paired comparison.

`research/task-10/profile_reference.py` separates MPO construction, JAX
lowering, XLA compilation, first execution, steady execution, StableHLO size,
and full `run_solution` time. Its sanitized output is stored under
`research/task-10/profiles/`.

### Candidate hypotheses

1. **Framework-native exact MPS plus whole-training scan (primary).**
   Convert TensorCircuit-NG's bond-2 CMZ coefficient chain into its equivalent
   diagonal MPO, apply it with `tc.MPSCircuit.apply_MPO`, and evaluate the
   43 local TFIM terms from the framework-owned MPS tensors with
   `tc.backend` contractions. Run the 200 Adam updates in one backend scan.
   This should replace generic path search with deterministic low-rank linear
   contractions while retaining exact TensorCircuit-NG quantum semantics.
2. **Scan-only ablation.** Keep the expert tensor-network objective unchanged
   and replace the host loop with `tc.backend.jaxy_scan` / JAX `lax.scan`.
   Expected benefit is limited to dispatch and history stacking; measure it to
   distinguish control-flow gain from representation gain.
3. **Fused single-qubit rotations.** Compose each `RX`, `RZ`, and `RY` into a
   single 2x2 TensorCircuit gate before MPS application. This preserves the
   exact ordered unitary and reduces MPS method calls and traced operations.
4. **Contractor-budget tuning.** Measure smaller OMECo search budgets only as
   an expert-path ablation. Keep it only if correctness, compilation, and
   steady execution improve together.
5. **Dense state plus matrix-free TFIM MVP (low priority).** The upstream
   Task 10 prompt mentions `PauliStringSum2MVP`, but a 22-qubit dense state has
   `2^22` complex amplitudes and prior evidence is much slower. Do not repeat
   without a materially new memory/compile argument.

An MPS candidate must not truncate singular values or cap the bond dimension.
Any approximation, changed layer, skipped update, hard-coded output, direct
NumPy/JAX simulator, or evaluator-specific shortcut is invalid.

## Frozen paired measurement and statistics plan

This plan is frozen before candidate timing:

- benchmark the immutable reference and candidate in the same Docker
  container and image, using a fresh evaluator process for every cell;
- use the canonical public Task 10 configuration and seed 2040;
- run at least six matched pairs for repository promotion, alternating
  `reference -> candidate` and `candidate -> reference`;
- additionally report the first five pairs' arithmetic means to satisfy the
  requested five-run comparison;
- apply a hard 300-second timeout to each evaluator process;
- count a runtime only when the evaluator prints a positive runtime and
  `Overall: PASS`;
- record image ID, host fingerprint, CPU/memory limits, source/evaluator
  hashes, all cell timings, means, medians, sample standard deviations,
  standard errors, percentage improvement, and paired speedups.

For eligible pairs `R_i, C_i`, use `S_i = R_i / C_i`. The predeclared
confidence interval is the two-sided 95% Student-t interval for the arithmetic
mean paired speedup:

`mean(S) +/- t_(0.975,n-1) * sample_stdev(S) / sqrt(n)`.

At six pairs, `t_(0.975,5) = 2.5705818366`. Promotion requires all validity
gates, six eligible pairs, lower candidate mean and median, wins in at least
80% of pairs, and a paired-speedup confidence lower bound above 1.0.

One environment deviation is already known: the current Colima allocation is
6 CPUs and 7.74 GiB, below the repository default 8 CPUs and 9 GiB. A campaign
run may use a separately recorded 6-CPU/7-GiB Docker profile for both sides,
but it must not be pooled with the default profile or called the default
baseline. The locked image build must succeed before formal pinned-environment
promotion; results from the newer ORBIT-Q development image are exploratory.

## Open evidence gaps

- The pinned image build was attempted after a network interruption and failed
  during `apt-get update`; no candidate had been edited.
- No global matched-hardware SOTA comparator is known for this exact workload.
- The primary exact-MPS hypothesis has not yet been implemented or timed.
- Scaling beyond the canonical two-layer configuration is out of claim scope
  unless an explicit public scale dataset is added before measurement.

## Post-freeze campaign outcome

This section records the result without changing the predeclared hypotheses or
statistics plan above.

The exact-MPS hypothesis passed as experiment `e02-noqr-mps` after removing
unnecessary QR/RQ canonicalization from the exact bond-dimension-four path.
The first five matched pairs measured a `4.892709x` ratio-of-means speedup and
`79.561424%` runtime reduction; all cells passed. The sixth predeclared pair
also passed, and the six-pair mean speedup confidence interval excluded 1.0.
See [`IMPLEMENTATION_COMPARISON.md`](IMPLEMENTATION_COMPARISON.md).

At the user's direction, the final relative comparison used the latest
available TensorCircuit nightly image for both sides rather than waiting for
the older exact-lock image. It ran with a symmetric 6-CPU/7-GiB container
profile because the Docker backend cannot supply the repository's requested
8 CPUs. This supports the same-machine relative claim but is not labeled as
the repository's default pinned-environment baseline or as global SOTA.
