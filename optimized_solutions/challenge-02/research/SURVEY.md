# ORBIT-Q Task 02 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-02`

Survey freeze: `2026-07-29T13:31:00Z`

Reference commit: `d13e2591574cc1480507b00bcb33b0c6a48e6b99`

This campaign covers only Task 02. The immutable expert, public canonical
workload, semantic constraints, optimization hypotheses, and measurement rule
are frozen before the first candidate edit.

## Evidence and claim boundary

The immutable human expert is `references/task-02/solution_2.py`
(`sha256:cd5776dfb223924edd83795bd8222032351008a584b55d747110521c709dcdd3`).
The public contract and evaluator are `tasks/task-02/problem.md`
(`sha256:e3e9b7517cd8aa1d0fad4fe4bce13cd61cde94d94c21b0324df95e80dd581a39`)
and `tasks/task-02/evaluator/evaluate_2.py`
(`sha256:0d9360d4812d033d364253dd478444f68918f1b05f04146eddcb846a3dca3a1e`).

At `2026-07-29T13:30Z`, the six open PRs on `sxzgroup/ORBIT-Q` were
repository infrastructure, agent-axis results, and the Task 07 design report;
none targeted Task 02. The separate benchmark repository likewise had no
active Task 02 optimization PR.

The immutable reference passed six fresh evaluator processes at
`4.559075`, `4.986945`, `4.990731`, `5.188701`, `4.813289`, and
`6.035628` seconds. Its mean is `5.095728` seconds, median `4.988838`
seconds, sample standard deviation `0.506494` seconds, and standard error
`0.206775` seconds. No external result uses this exact evaluator, target
profile, initialization, 500-step Adam trajectory, image, and hardware
allocation. Claims are limited to same-machine gains over the bundled expert
on the canonical public workload.

## Framework and environment

Measurements use image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
with six CPUs, 7 GiB memory, no network, and a fresh evaluator process per
cell. The dependency lock is
`envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
No TensorCircuit downgrade is used.

| Component | Version | Inspected source |
| --- | --- | --- |
| TensorCircuit-NG | `1.8.0.dev20260726` | `quantum.py` (`sha256:fcaee21ba5ccde1b89c46e2f5424c48d342e3bfaedf72adba672ab6bd4ded703`), `circuit.py` (`sha256:5c4d569325369d957dc60bbeca8a581508549ff9813a7a163de61a6294864662`), and `gates.py` (`sha256:6db54dab1390360273ef527a1fadc4f36ba169d7e961dcc4bd049b6b2a43cbe4`) |
| JAX / JAXLIB | `0.10.0` / `0.10.0` | TensorCircuit JAX backend (`sha256:88657aebf8e5d566ac4e653abe327083da0253f02a3a297a134b871ffe4baab9`) |
| Optax | `0.2.8` | Adam with the required `0.015` learning rate and exactly 500 updates |
| TensorNetwork / Quimb | `0.5.1` / `1.11.1` | Installed support stack; the dense state path does not invoke OMECo |

TensorCircuit documents sparse, dense, vmap, termwise, and MPO alternatives
for Pauli-sum expectation values:
<https://tensorcircuit.readthedocs.io/en/latest/whitepaper/6-2-pauli-string-expectation.html>.
Its public quantum source documents both `PauliStringSum2MVP` and
`PauliStringSum2COO`, and implements order-2 Renyi entropy as
`-log(trace(rho @ rho))`:
<https://tensorcircuit.readthedocs.io/en/latest/_modules/tensorcircuit/quantum.html>.
TensorCircuit's FAQ describes the same reduced-density-matrix and Renyi APIs
used by the expert:
<https://tensorcircuit.readthedocs.io/en/stable/faq.html>.
JAX documents that `lax.scan` lowers a fixed loop to one WhileOp rather than
unrolling it:
<https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>.

## Task 02: entanglement-profile-constrained VQE

The expert evolves a dense 12-qubit complex64 state from the Neel basis state.
Each of three blocks applies one even and one odd brickwork sublayer. Every
sublayer applies independent `RY -> RZ` rotations, followed by independent
`RXX -> RYY -> RZZ` gates on its bonds. After each complete block it computes
the left-half Renyi-2 entropy. The final state is evaluated against the
45-term open XXZ plus staggered-field Hamiltonian. Reverse-mode differentiation
updates all 243 independent float32 parameters with Adam for exactly 500 steps.

Every viable candidate must preserve:

- the 12-qubit Neel input, six trainable sublayers, and all 243 independent
  parameters initialized from the fixed NumPy seed 2026;
- the exact TensorCircuit gate convention and every ordered
  `RY -> RZ -> RXX -> RYY -> RZZ` operation;
- the normalized state after each complete block and the three half-chain
  order-2 Renyi entropies;
- the exact 45-term XXZ/staggered-field energy, entropy weight `0.25`, target
  profile `[0.30, 0.60, 0.80]`, and complex64 semantics;
- exactly 500 sequential Adam updates and every pre-update energy, loss,
  entropy MSE, and three-entropy row;
- TensorCircuit-NG as the central quantum framework and the original four-key
  NumPy output contract.

## Source-supported cost model

One loss contains 243 parameterized TensorCircuit gates, three 64-by-64
reduced-density matrices, three dense `rho @ rho` products, and one
45-term Hamiltonian action on 4,096 amplitudes. Reverse mode differentiates
through all of these operations. The expert stages one update once, but then
dispatches the compiled update from Python 500 times.

The installed `PauliStringSum2MVP` loops over all 45 Pauli terms in Python
during tracing and emits a separate mask/flip/multiply/accumulate path per
term. `PauliStringSum2COO` instead creates one native backend sparse operator.
The installed pure-state reduced-density-matrix path reshapes the amplitude
vector to 64-by-64 and forms `rho = w @ w.adjoint()`. For a Hermitian density
matrix, `trace(rho @ rho) = sum(abs(rho) ** 2)`, so the second dense matrix
product is algebraically removable without changing the requested entropy.

The three Pauli rotations on a bond commute. Their product is one exact
parity-preserving 4-by-4 TensorCircuit gate. Likewise, each `RY -> RZ`
sequence is one exact 2-by-2 gate. This can reduce 81 gate applications per
block to 35 while keeping every angle independent.

## Frozen experiments

1. **Whole-training TensorCircuit scan.** Carry parameters and Optax state
   through `K.jaxy_scan` and return all pre-update observables. This tests host
   dispatch independently.
2. **Exact Renyi-2 purity reduction.** Keep TensorCircuit's reduced density
   matrix but replace `trace(rho @ rho)` with `sum(abs(rho) ** 2)`. Require
   state, entropy, loss, gradient, one-update, and short-history agreement.
3. **Native sparse XXZ action.** Compare TensorCircuit
   `PauliStringSum2COO` plus backend sparse-dense multiplication against the
   expert's termwise MVP. Retain only if canonical timing improves.
4. **Exact local gate fusion.** Replace each parameterized `RY -> RZ` pair
   and commuting `RXX -> RYY -> RZZ` triple with its exact differentiable
   TensorCircuit unitary. Audit random gate matrices and end-to-end gradients.
5. **Initial-state constant construction.** Build the same Neel amplitude
   vector without eight circuit nodes only if profiling shows a measurable
   residual; this is expected to be negligible.
6. **MPS or truncated Schmidt simulation.** Reject for the canonical
   campaign: the entropy checkpoint and unconstrained entanglers do not give a
   static small exact bond bound, and truncation would change semantics.

Each retained factor will receive six alternating matched pairs against its
accepted parent, plus one chart. The final combination will be remeasured
against the immutable expert.

## Correctness and measurement rule

Before canonical timing, compare reference and candidate block states,
entropies, energy, complete loss, gradient leaves, one Adam update, and a
short physical history. Frozen complex64 tolerances are `2e-6` for gates,
states, observables, and initial loss; `1e-5` for gradient elements; and
`2e-4` for post-update and short-history physical outputs. Raw Adam parameter
differences caused only by theoretical-zero complex64 gradients must be
reported and cannot replace the physical-output checks.

Final evidence uses six counterbalanced canonical pairs in one no-network
container. Odd pairs run reference then candidate; even pairs reverse order.
Every cell is a fresh evaluator process with a 300-second cap. Promotion
requires all 12 cells to pass, candidate wins at least five of six pairs,
candidate mean and median are lower, and the lower endpoint of a two-sided
95% Student-t interval for mean pairwise speedup exceeds `1.0`.
