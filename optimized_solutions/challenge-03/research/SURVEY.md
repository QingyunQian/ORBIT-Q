# ORBIT-Q Task 03 Runtime Optimization Survey

**Status: READY**

Campaign task: `task-03`

Survey freeze: `2026-07-29T12:38:00Z`

Reference commit: `d13e2591574cc1480507b00bcb33b0c6a48e6b99`

This campaign covers only Task 03. The immutable expert, canonical public
workload, semantic constraints, hypotheses, and promotion rule were frozen
before the first candidate edit.

## Evidence and claim boundary

The immutable expert is `references/task-03/solution_3.py`
(`sha256:a451480b08610e24072098171ac57364efc764ddc5e77e3d95028f009b6d5c89`).
The contract is `tasks/task-03/problem.md`
(`sha256:c4e3f6122d85b28f1745857a6aed29f73660ca8b0106a5c17ba500e5f5f1d5fb`)
and the evaluator is `tasks/task-03/evaluator/evaluate_3.py`
(`sha256:b0a173f181857735b94fea3a9d03e11b595d4584b2a4bc33da80973e4b5c6d36`).

Six fresh-process reference runs passed at `4.262369`, `4.044382`,
`4.064228`, `4.297515`, `4.053138`, and `4.200180` seconds. Their mean is
`4.153635` seconds, median `4.132204` seconds, and standard error `0.046447`
seconds. No external benchmark uses this exact seeded optimizer trajectory and
evaluator, so claims are limited to a same-host gain over the bundled expert.

At survey time the benchmark repository had no open Task 03 optimization PR.
Open work on other task IDs is outside this one-task campaign.

## Framework and environment

Measurements use image
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
with 6 CPUs, 7 GiB, no network, and a 300-second cap per fresh evaluator
process. The dependency lock is
`envs/tensorcircuit-py311/requirements.lock`
(`sha256:cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b`).
No framework downgrade is used.

| Component | Version | Inspected API/source |
| --- | --- | --- |
| TensorCircuit-NG | `1.8.0.dev20260726` | `tensorcircuit/circuit.py:BaseCircuit.mid_measurement`, `tensorcircuit/gates.py:rxx_gate,rzz_gate,rx_gate` |
| JAX / JAXLIB | `0.10.0` / `0.10.0` | TensorCircuit `JaxBackend.jit`, `value_and_grad`, `vmap`, and `jaxy_scan` |
| TensorNetwork | `0.5.1` | circuit tensor nodes and contraction backend |
| Optax | `0.2.8` | exactly 300 Adam updates at learning rate `0.01` |
| Quimb | `1.11.1` | installed but not used by the expert |

TensorCircuit documents JAX-backed automatic differentiation, JIT, and VMAP as
its standard variational-programming path:
<https://tensorcircuit.readthedocs.io/en/stable/quickstart.html>.
The backend API documents `scan`, `vmap`, and the extended JAX backend:
<https://tensorcircuit.readthedocs.io/en/latest/api/backends/jax_backend.html>.
JAX documents that `lax.scan` lowers a fixed loop to one WhileOp:
<https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html>.

## Task 03: probability-aware post-selected cooling

The expert starts in `|+>^12`, alternates five even-bond and five odd-bond
layers, and after every layer post-selects all six even-indexed qubits on
computational outcome zero. Every layer has independent `RXX`, `RZZ`, and
single-qubit `RX` parameters. It differentiates the final TFIM energy plus a
penalty derived from all 60 conditional measurement probabilities and performs
exactly 300 sequential Adam updates.

Every candidate must preserve:

- 12 logical qubits, ten alternating layers, all 230 independent float32
  parameters, seed 2027, and complex64 TensorCircuit semantics;
- the gate order `RXX -> RZZ -> RX`, all zero-outcome post-selections, and
  normalization after each measurement event;
- all 60 event probabilities in the objective, their product in the success
  history, and the mean of their logs in both loss and output;
- the open-boundary TFIM with transverse field `0.9`;
- exactly 300 Adam updates and pre-update histories;
- all four NumPy outputs with their original shapes and meanings.

## Exact structure and expected bottleneck

After a post-selection layer, every even-indexed qubit is exactly `|0>`. Each
subsequent brickwork gate couples one even qubit to at most one odd qubit.
Because no layer couples two surviving odd qubits, the post-selected state is
always

`|0>_0 tensor |psi_1> tensor |0>_2 tensor ... tensor |psi_11>`.

The initial `|+>^12` is also a product. Consequently each active two-qubit
gate followed by the zero projection is an exact differentiable 2-by-2 map on
one surviving odd qubit. The six event probabilities factor, and the TFIM
expectation needs only six one-qubit X/Z expectations. This is a
challenge-structure reduction, not an approximation or rank truncation.

The expert nevertheless traces ten 12-qubit circuit states, 60 separate
projector insertions, and a 23-term Hamiltonian MVP inside every differentiated
loss. Compilation is expected to dominate the roughly four-second end-to-end
runtime. A product-state conditional-map formulation can reduce state storage
from 4096 amplitudes to twelve amplitudes and sharply shrink the traced graph.

## Frozen factors

1. **All-projectors-at-once.** On the full 12-qubit state, insert the six
   commuting zero projectors in one circuit, normalize once, and use
   `log(||P state||^2)/6`. The sum of sequential conditional log probabilities
   telescopes exactly to `log(||P state||^2)`, and no returned quantity exposes
   individual event logs.
2. **Whole-training TensorCircuit scan.** Replace 300 host dispatches with
   `K.jaxy_scan` while preserving pre-update histories and the final update.
3. **Exact product-state conditional maps.** Contract each two-qubit
   TensorCircuit gate with the fixed even-qubit input and zero-output tensors,
   normalize each of the six surviving one-qubit states, and evaluate the
   product-state TFIM analytically with TensorCircuit backend operations.
4. **Vectorize the six local maps.** Test `K.vmap` only after the scalar exact
   reduction is validated; discard it if the tiny batch increases compilation.
5. **Gate-matrix algebra.** Reuse TensorCircuit `rx_gate`, `rxx_gate`, and
   `rzz_gate`; direct trigonometric reimplementation is lower fidelity and is
   not the primary path.

## Correctness and measurement rule

The exact-reduction audit compares:

1. projected normalized states and total probabilities on deterministic
   reduced instances;
2. initial energy, success, mean log probability, and loss;
3. every gradient leaf;
4. one non-degenerate Adam update;
5. short complete histories;
6. the canonical 300-update evaluator.

Frozen complex64 tolerances are `2e-6` for state and observable comparisons,
`2e-5` for gradients and one-update parameters, and `2e-4` for short histories.
The canonical evaluator remains the ultimate correctness gate.

Each important factor receives six alternating matched pairs. Final evidence
uses six reference/candidate pairs in one no-network container; odd pairs run
reference then candidate and even pairs reverse the order. Promotion requires
all 12 cells to pass, candidate wins at least five pairs, lower mean and median,
and a two-sided 95% Student-t lower bound for mean pairwise speedup above `1`.
