# ORBIT-Q Task 01 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-01`

Survey freeze: `2026-07-29T14:59:00Z`

Reference commit: `0819ed34dd3f1eaa8f77587c2a40d95a420ea829`

This campaign starts from the immutable human expert. The Task 01 MPO solution
already present on `main` is historical evidence, not an accepted parent or a
constraint on the search. The canonical public workload record and six-run
immutable-reference validation were completed before candidate editing.

## Evidence and claim rules

The immutable comparator is
`references/task-01/solution_1.py`
(`sha256:80011df1b68a009b489982a8f5c1ec47f693a187f923a5c906fb2b6f4d51c2a1`).
The problem is `tasks/task-01/problem.md`
(`sha256:4c3761842d2fc7e0e9ce1d825308fc32012ba7470d7ea6e17d53f782238a6b2d`)
and the timer/contract are implemented by
`tasks/task-01/evaluator/evaluate_1.py:evaluate`
(`sha256:dd143d07e9e9f7298f9263b5d34346c816e293cc5eccc9f80177f90bb608e63e`).

No public paper supplies a matched Task 01 evaluator, implementation, package
set, and hardware allocation. Therefore the final report may claim a paired
speedup over the bundled human expert, but not a global SOTA runtime.

## Pinned execution environment

Measurements use image
`orbitbreakers-expert-benchmarks:tensorcircuit-py311`, image ID
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`,
with 6 CPUs, 7 GiB memory, no network, and a 300-second timeout. The live image
contains TensorCircuit-NG `1.8.0.dev20260726`, TensorNetwork-NG `0.5.1`, JAX and
JAXLIB `0.10.0`, Optax `0.2.8`, Quimb `1.11.1`, and OMECo `0.2.0`.

Inspected framework files:

- `tensorcircuit/mpscircuit.py`
  (`sha256:a1fb60ad915a86e498f3234769dc2461fcccc495505b138cefc49b6ca4f55655`);
- `tensorcircuit/templates/measurements.py`
  (`sha256:7a69043bd81745254ab106f3cdd0911720fdb91e0be70b784f86c96fbcfaa615`);
- `tensorcircuit/quantum.py`
  (`sha256:fcaee21ba5ccde1b89c46e2f5424c48d342e3bfaedf72adba672ab6bd4ded703`);
- `tensorcircuit/backends/jax_backend.py`
  (`sha256:88657aebf8e5d566ac4e653abe327083da0253f02a3a297a134b871ffe4baab9`).

TensorCircuit documents MPS/MPO input and expectation APIs in its
[quick start](https://tensorcircuit.readthedocs.io/en/latest/quickstart.html),
and the framework implementation exposes exact `MPSCircuit` gate application
and MPO expectation. The
[TFIM representation tutorial](https://tensorcircuit.readthedocs.io/en/stable/tutorials/tfim_vqe_diffreph.html)
motivates comparing Pauli-sum and MPO Hamiltonians. JAX documents
[`lax.scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html) as
a single loop primitive and explains synchronization requirements in its
[benchmarking guide](https://docs.jax.dev/en/latest/benchmarking.html) and
[asynchronous dispatch note](https://docs.jax.dev/en/latest/async_dispatch.html).
General exact MPS time evolution and its bond-dimension tradeoffs are reviewed
by Paeckel et al.,
[arXiv:1901.05824](https://arxiv.org/abs/1901.05824), while Schollwöck's
[MPS review](https://arxiv.org/abs/1008.3477) supplies broader context. These
sources support mechanisms, not matched runtime claims.

## Task 01: DMRG-MPS variational refinement

The evaluator generates and normalizes a 32-site Quimb DMRG MPS of maximum
bond dimension 8 before timing. Inside the timed region the expert converts
that state to a TensorCircuit `QuOperator`, creates a regular
`tc.Circuit(..., mps_inputs=...)`, and applies four alternating brickwork
layers. Every layer contains `RZ -> RY -> RZ` on all sites. Even layers act on
16 even bonds and odd layers on 15 odd bonds; each active bond receives
`RXX -> RYY -> RZZ`.

There are 570 independent float32 parameters: 384 one-qubit angles and 186
two-qubit angles. Initialization is NumPy PCG64 seed 1234 with standard
deviation `1e-4`. The scalar loss is the 63-term open TFIM Hamiltonian,
`-sum(ZZ) - 1.05 sum(X)`, evaluated through 63 separately specified
TensorCircuit measurements. The expert performs exactly 500 Adam updates at
learning rate `0.005` and records the pre-update energy for each update.

Every candidate must preserve:

- the evaluator-provided DMRG state and its qubit/site order;
- all 570 independent parameters, initialization bytes, gate order, and
  complex64 TensorCircuit semantics;
- four exact unitary layers with no truncation or rank cap;
- exactly 500 Adam updates and the pre-update history definition;
- the sole returned key `energy_history`, with NumPy-compatible length 500;
- the immutable expert docstring byte-for-byte.

Changing the optimizer, update count, initialization, ansatz, DMRG input,
precision, Hamiltonian, returned history, or evaluator is prohibited.

## Cost model and profiling questions

The regular-circuit representation avoids a dense `2^32` state, but it creates
a general tensor network containing the input MPS and hundreds of gate nodes.
Each gradient therefore differentiates through circuit construction,
contraction-path processing, a full contraction, and 63 observable
contractions. The four-layer brickwork circuit also admits an exact MPS
description. Each nearest-neighbor two-qubit unitary has operator Schmidt rank
at most four; since a physical bond is acted on twice, the worst exact bond
bound is `8 * 4^2 = 128`. This is small enough to test an exact, untruncated
framework-native MPS path.

The profile must separate:

1. objective tracing/compilation from steady update execution;
2. circuit propagation from Hamiltonian expectation;
3. 63 Pauli contractions from one bond-dimension-three TFIM MPO;
4. host dispatch from compiled gradient work;
5. general-network contraction/path overhead from exact local MPS kernels.

End-to-end promotion timing includes all setup, compilation, 500 updates,
device synchronization, and NumPy result conversion.

The immutable-reference baseline
`results/task-01-reference-20260729/results.json`
(`sha256:61aca162f2fd80634f253464e885d83ca16bbb7ffefaa99ade2e83b23412e8c2`)
contains six passing runs: `54.896941`, `56.031343`, `52.621569`,
`55.264405`, `58.445563`, and `56.618919` seconds. Mean runtime is
`55.6464567 s`, median `55.647874 s`, sample standard deviation
`1.9383756 s`, and standard error `0.7913385 s`. This baseline establishes
the reference distribution; a speedup claim still requires alternating
matched candidate pairs.

## Predeclared single-factor hypotheses

Each retained factor is first validated on the same evaluator-generated DMRG
state and then benchmarked independently. A combined candidate may contain only
factors whose individual effect and interaction are reported.

1. **Direct TFIM MPO:** replace the 63 `parameterized_measurements`
   contractions with an exact bond-dimension-three TensorCircuit MPO
   expectation. The historical main-branch experiment reported about 1.53x
   under a different image and only three pairs, so it must be remeasured.
2. **Whole-training backend scan:** use `K.jaxy_scan` for all 500 Adam updates,
   preserving every pre-update energy, to reduce Python dispatch and expose the
   whole trajectory to one outer JIT.
3. **Fused Euler site gate:** multiply the three native TensorCircuit
   `RZ -> RY -> RZ` matrices and apply one differentiable 2x2 tensor per site.
4. **Fused commuting bond gate:** multiply native `RXX`, `RYY`, and `RZZ`
   matrices into one exact 4x4 unitary per active bond.
5. **Exact TensorCircuit MPS propagation:** operate on the converted input MPS
   with `tc.MPSCircuit`, no truncation, and compare its QR/SVD cost against the
   general circuit.
6. **Static exact local-MPO propagation:** express each fused two-site gate in
   the `{I⊗I, X⊗X, Y⊗Y, Z⊗Z}` operator basis and contract its exact rank-four
   MPO into adjacent MPS tensors using TensorCircuit backend operations. No
   SVD, canonicalization, truncation, or raw dense-state simulator is allowed.
7. **Direct MPS TFIM environment:** contract the exact propagated MPS against
   the bond-three TFIM MPO with a left-to-right TensorCircuit-backend
   environment instead of invoking a generic global path search.
8. **Contractor/path-search variants:** compare OMECo and supported deterministic
   greedy/preprocessing paths only where the generic tensor-network candidate
   remains competitive.
9. **Static input tensor extraction:** convert and canonicalize the supplied
   Quimb MPS once outside the transformed objective, then close over immutable
   TensorCircuit tensors.
10. **Parameter/tree layout:** compare the expert flat length-570 array with a
    layout that exposes static layer offsets without changing element order or
    Adam state.
11. **Rematerialization:** test TensorCircuit/JAX checkpoint boundaries only if
    memory pressure is measured; discard if recomputation loses runtime.
12. **Best-factor consolidation:** combine only validated factors, rerun
    equivalence, and conduct leave-one-factor-out ablations to expose
    interactions and the dominant contribution.

An attractive but invalid shortcut is to return the supplied DMRG energy or
skip the variational computation because the evaluator tolerance is loose. The
campaign will not exploit that scoring loophole: the complete 570-parameter,
four-layer, 500-update expert computation remains intact.

## Frozen measurement protocol

- Functional checks: identical shared DMRG input; compare initial energy,
  gradients, one Adam update, reduced histories, and the official evaluator.
- Numerical tolerance: report maximum absolute/relative differences and use
  the official evaluator as the final authority; complex64 reassociation is
  allowed only when every official run passes.
- Factor screening: at least five fresh timed executions per viable factor.
- Promotion: six alternating matched pairs in one pinned environment; odd
  pairs run reference then candidate, even pairs reverse the order.
- A promoted candidate must pass 6/6, lower both mean and median runtime, win
  at least 5/6 pairs, and have a Student-t 95% lower confidence bound for the
  paired speedup above 1.
- Every retained factor receives its own ablation figure. The final report
  includes raw runtimes, uncertainty, correctness evidence, source hashes, and
  all discarded or negative results.

## Live upstream overlap check

At `2026-07-29T14:59:00Z`, the open PRs on `sxzgroup/ORBIT-Q` were #3, #4,
#5, #6, and #7. None is an active Task 01 human-expert runtime optimization
campaign, so no overlapping implementation was found.
