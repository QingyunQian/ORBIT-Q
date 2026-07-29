# ORBIT-Q Task 06 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-06`

Survey freeze: `2026-07-29T01:36:02Z`

Reference commit: `5af98f27b9404c513df8eee0f4568b1512edee19`

This campaign covers only Task 06. The immutable expert, public canonical
workload, semantic constraints, hypotheses, and measurement rule are frozen
before the first candidate edit.

## Evidence and claim boundary

The immutable human expert is `references/task-06/solution_6.py`
(`sha256:0e7fec8d11135241eb3f3501f3651f3f337e08c636407b3da8a2858c2b3d85d1`).
The public contract and evaluator are `tasks/task-06/problem.md`
(`sha256:e3e8a8044c322027327b6e694dbb476f67dbc09465ea80dd4f20aebf8c64c8a7`)
and `tasks/task-06/evaluator/evaluate_6.py`
(`sha256:0d2dfc7f30087896fb599925f9110190a3a61358263688dbb09cc36115a23998`).

The old bootstrap did not produce a Task 06 runtime because its then-installed
framework lacked the required raw-mode ODE API. The current latest
TensorCircuit-NG image runs the immutable expert successfully. Six complete
fresh-process baselines are `43.976600`, `45.097434`, `44.415753`,
`44.864424`, `45.481100`, and `46.387673` seconds: mean `45.037164` seconds,
median `44.980929` seconds, and standard error `0.344740` seconds.

No external result uses this exact evaluator, seeded initialization, optimizer
trajectory, ODE tolerances, container, and hardware allocation. This campaign
may claim only a paired gain over the bundled expert and a repository/campaign
best, not a global hardware-independent SOTA result.

At `2026-07-29T01:26Z`, the upstream repository had no open pull request
matching Task 06. The campaign therefore does not duplicate an active Task 06
optimization.

## Framework and environment

Measurements use Docker image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
with six CPUs, 7 GiB memory, no network, and a fresh evaluator process for
every cell. The tracked lock is
`envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
Per maintainer direction, no TensorCircuit downgrade is used.

| Component | Version | Inspected source |
| --- | --- | --- |
| TensorCircuit-NG | `1.8.0.dev20260726` | `quantum.py` (`sha256:fcaee21ba5ccde1b89c46e2f5424c48d342e3bfaedf72adba672ab6bd4ded703`), `timeevol.py` (`sha256:64013d6bed57c842f00df254ce84d35c70b25c2a01f5d774b8d503d2e7a097d7`), and `circuit.py` (`sha256:5c4d569325369d957dc60bbeca8a581508549ff9813a7a163de61a6294864662`) |
| JAX / JAXLIB | `0.10.0` / `0.10.0` | TensorCircuit JAX backend (`sha256:88657aebf8e5d566ac4e653abe327083da0253f02a3a297a134b871ffe4baab9`) |
| Diffrax | `0.7.2` | called through `tc.timeevol.ode_evol_global` with `Tsit5`, PID tolerance control, and `max_steps=16` |
| Optax | `0.2.8` | exactly 100 Adam updates at learning rate `0.12` |

TensorCircuit's ODE documentation describes full-system sparse Hamiltonians as
the intended efficient input to global ODE evolution:
<https://tensorcircuit.readthedocs.io/en/stable/api/experimental.html>.
The installed `ode_evol_global` source confirms that raw mode delegates to
Diffrax and preserves the configured `rtol`, `atol`, and `max_steps`.
Diffrax documents `diffeqsolve`, adaptive step-size control, maximum steps,
and reverse-mode checkpointing:
<https://docs.kidger.site/diffrax/api/diffeqsolve/> and
<https://docs.kidger.site/diffrax/api/adjoints/>.
JAX documents that `lax.scan` lowers a fixed loop to one WhileOp and that
host conversion synchronizes asynchronous computation:
<https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html> and
<https://docs.jax.dev/en/latest/async_dispatch.html>.

## Task 06: digital-analog hybrid VQE

The expert prepares the 14-qubit Neel state and applies four hybrid blocks.
Each block performs an actual differentiable continuous-time ODE evolution
under

`J_l * sum_i (X_i X_(i+1) + Y_i Y_(i+1)) + Delta_l * sum_i (-1)^i Z_i`

with trainable bounded time, coupling, and detuning, followed by independent
`RZ -> RY -> RZ` rotations on all 14 qubits. The loss is the energy density of
the specified 53-term target Hamiltonian. The expert differentiates this
four-block computation and performs exactly 100 sequential Adam updates,
recording every pre-update energy.

Every viable candidate must preserve:

- all 14 qubits, four analog ODE solves per loss, and the exact Neel state;
- actual continuous-time ODE integration through TensorCircuit with
  `rtol=atol=1e-6` and `max_steps=16`, never a Trotter/product formula;
- all 180 trainable float32 values: 4 times, 4 couplings, 4 detunings, and
  `4 * 14 * 3 = 168` independent digital rotation angles;
- the `sigmoid`/`tanh` parameter maps, seed 2026 initialization, target
  Hamiltonian, complex64 computation, 100 Adam updates, and pre-update history;
- all four required NumPy outputs with their original meanings and shapes.

## Source-supported bottleneck

Each Diffrax vector-field evaluation calls two separate
`PauliStringSum2MVP` closures. In TensorCircuit-NG 1.8, that helper implements
every Pauli term as a reshape, broadcast mask, slice/flip, multiply, and
accumulate. The XY operator has 26 terms and the field has 14 terms, so each
adaptive Runge-Kutta stage executes 40 termwise full-state transforms; reverse
mode differentiates through the same work. Four ODE solves occur per loss and
100 loss/gradient/update steps are required.

The framework's `PauliStringSum2COO` instead constructs a native JAX sparse
operator, and the JAX backend exposes `sparse_dense_matmul`. The installed
TensorCircuit ODE docstring explicitly recommends a sparse full-system
Hamiltonian for efficiency. This is the primary hypothesis.

The 168 digital gates are a secondary trace/compile and execution cost. Each
qubit's exact Euler sequence can be represented by one differentiable 2x2
TensorCircuit gate without tying or removing any angle. The Python host loop
dispatches one jitted optimizer step 100 times and is a lower-risk secondary
target.

## Frozen experiments

Each experiment starts from the latest accepted candidate and must pass a
public reduced screen before canonical promotion.

1. **Native sparse Hamiltonian action.** Build the analog and target
   Hamiltonians with TensorCircuit `PauliStringSum2COO`, apply them with the
   TensorCircuit JAX backend's sparse-dense multiplication, and keep
   `ode_evol_global(..., mode="raw", ode_backend="diffrax")`. Retain only if
   a canonical run passes and improves on the `45.037164`-second reference
   mean.
2. **Exact digital Euler fusion.** Form one differentiable 2x2 gate per qubit
   equal to the expert's ordered `RZ -> RY -> RZ` sequence and apply it through
   TensorCircuit. Require initial-energy and gradient agreement at complex64
   tolerance before runtime promotion.
3. **Whole-training TensorCircuit scan.** Carry parameters and Optax state
   through `K.jaxy_scan` for exactly 100 updates and return all pre-update
   energies. Retain only if canonical runtime improves over the accepted
   predecessor.
4. **ODE solver selection.** Compare TensorCircuit's available `Tsit5`,
   `Dopri5`, and `Dopri8` only after higher-value structural changes. All
   candidates must retain the same tolerance and step bound and pass the
   canonical energy/output contract.
5. **Parameter-tree simplification.** Flatten the four-leaf parameter PyTree
   only if profiling shows optimizer/control overhead remains material.
6. **Direct algebraic or free-fermion evolution.** Reject for this campaign:
   although the analog Hamiltonian is quadratic, replacing the required ODE
   with a closed-form exponential, matchgate decomposition, or handwritten
   simulator would change the requested method/framework fidelity.

## Correctness and measurement rule

For numerical audits, compare reference and candidate initial energy,
gradient leaves, one Adam update, and short histories. Frozen complex64
tolerances are `5e-5` for initial energy, `5e-4` for maximum gradient element,
and `2e-3` for short physical energy history unless a candidate is algebraically
bit-equivalent.

Final evidence is six counterbalanced canonical pairs in one no-network
container. Odd pairs run reference then candidate; even pairs reverse order.
Every cell is a fresh evaluator process with a 300-second cap. Promotion
requires all 12 cells to pass, candidate wins in at least five of six pairs,
lower candidate mean and median, and a two-sided 95% Student-t lower bound on
mean pairwise speedup above `1.0`.
