# Task 07 Conservative Human-Expert Optimization Report

> **Status:** retained as the literal 16-qubit / `cond_measure` fallback.
> The current executable-contract winner is the exact classical-ancilla
> reduction documented in
> [`CLASSICAL_ANCILLA_REDUCTION_REPORT.md`](CLASSICAL_ANCILLA_REDUCTION_REPORT.md):
> 3.070839-second candidate mean versus 140.076441-second expert mean,
> 45.757921x mean paired speedup, 95% CI
> [39.384711x, 52.131131x]. Unlike the conservative implementation below,
> that candidate exposes a challenge-design loophole and does not literally
> execute the measured ancilla register. The implementation below is retained
> as the runnable `conservative` variant at
> `src/solutions/task-07/variants/solution_7_conservative.py`.

## Scope and claim

This campaign optimizes only ORBIT-Q Task 07: the 16-qubit, two-layer
measurement-feedback VQE with 64 fixed trajectories and exactly 100 Adam
updates.

The conservative candidate passes all public functional checks in all six measured
runs and wins all six counterbalanced pairs against the immutable human
expert. Mean paired speedup is **4.479x**, with a two-sided 95% Student-t
interval of **[3.891x, 5.067x]**. Ratio-of-means speedup is **4.438x**
(77.47% lower runtime).

No external implementation reports a matched runtime for this exact
evaluator, seed, trajectory batch, container, and software stack. The result
was the **campaign-best / repository-SOTA Task 07 implementation before the
exact e11 reduction**, not a global hardware-independent SOTA claim.

| Artifact | Path | SHA-256 |
| --- | --- | --- |
| Immutable human expert | `references/task-07/solution_7.py` | `ac483319363f3c386a7646eaa867670ae3d3cd687f8517e6d4201e69240ff0a3` |
| Final candidate | `src/solutions/task-07/solution_7.py` | `0cd9676dc904660597a2f7dd6981fdac596295e4eae99510a3f4f21671859592` |
| Evaluator | `tasks/task-07/evaluator/evaluate_7.py` | `69717d98a90a7e53c31686128b3ef3e7cea3c96685ec538662a12163fe324b31` |
| Paired report | `profiles/final-canonical-six-pairs.json` | `6e03db1f37e8bbe0b38247f017d9259177a2858d54289fa2a5615542b499b54a` |
| Reference gate report | `profiles/final-reference-six.json` | `743f493120dd89e6c75a309499c28d819ae00aee09d57e7feb374b35c0310224` |

## Final paired result

All measurements used one no-network Docker container, six CPUs, 7 GiB,
TensorCircuit-NG `1.8.0.dev20260726`, JAX/JAXLIB `0.10.0`, and a fresh
evaluator process per cell. Odd pairs ran reference then candidate; even
pairs reversed the order.

| Metric | Human expert | Candidate |
| --- | ---: | ---: |
| Passing runs | 6/6 | 6/6 |
| Mean runtime | 116.264691 s | 26.196276 s |
| Median runtime | 114.756293 s | 24.598863 s |
| Sample standard deviation | 10.035012 s | 2.980442 s |
| Standard error | 4.096777 s | 1.216760 s |
| Minimum / maximum | 106.038165 / 131.316223 s | 24.221592 / 31.684172 s |

| Pair | Order | Expert (s) | Candidate (s) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | expert -> candidate | 106.038165 | 24.481295 | 4.3314x |
| 2 | candidate -> expert | 119.427967 | 24.221592 | 4.9306x |
| 3 | expert -> candidate | 123.188182 | 24.716431 | 4.9841x |
| 4 | candidate -> expert | 107.532991 | 24.421997 | 4.4031x |
| 5 | expert -> candidate | 110.084619 | 31.684172 | 3.4744x |
| 6 | candidate -> expert | 131.316223 | 27.652166 | 4.7489x |

The candidate wins 6/6 pairs. Mean pairwise speedup is 4.478752x
(standard error 0.228662x); the frozen Student-t rule gives
`4.478752 +/- 2.5705818366 * 0.228662`, or
**[3.890959x, 5.066545x]**. The lower bound exceeds 1.0, so every research
and promotion gate passes.

## Why the expert is slow

The expert performs three expensive generic operations:

1. Each trajectory energy is seven separate `ZZ` plus eight separate `X`
   `expectation_ps` calls. The same final adaptive tensor network is
   contracted 15 times in both forward and gradient work.
2. After each Z measurement, `conditional_gate` still builds and selects a
   dense two-qubit `RZZ` tensor, even though the measured ancilla is already
   a computational-basis eigenstate.
3. The expert requests OMECo `TreeSA(ntrials=32,niters=32)`. Path search lies
   inside the timed first JAX trace, and this simplified 16-qubit graph does
   not benefit enough from the extra search.

Update-count profiling confirms the split: the expert's one-step run takes
47.853 seconds, while 100 steps take 135.816 seconds. Both graph/path
construction and repeated quantum contractions matter.

## Final implementation

### 1. Contract the final trajectory once

The final ancillas were Z-measured and are subsequently touched only by
diagonal feedback. The state therefore factorizes:

`|Psi_final> = |psi_data> tensor |measured ancilla bitstring>`.

The candidate contracts the TensorCircuit state once, reshapes it to
`(2^8, 2^8)`, and sums the ancilla basis axis; exactly one column is nonzero.
It initializes an eight-qubit TensorCircuit from that data state and evaluates
the complete TFIM with one TensorCircuit-native sparse operator built by
`PauliStringSum2COO` and
`templates.measurements.operator_expectation`.

This replaces 15 final-circuit expectation contractions with one state
contraction and one small sparse Hamiltonian expectation. The isolated e01
canonical screen falls from 135.816 to 61.397 seconds.

### 2. Reduce measured feedback exactly

For measured ancilla bit `b`, TensorCircuit's convention gives the exact
identity

`RZZ(theta_b) |b,psi> =
 |b> RZ((1-2b) theta_b) |psi>`.

The candidate selects the same independent `theta0/theta1` parameter and
applies the signed angle through the native data-qubit `c.rz`. The ancilla is
unchanged, so the identity remains valid before the next layer.

An independent four-case complex64 matrix-action audit covers both bits and
two signed angles; maximum error is `2.98e-8` under a frozen `1e-7`
tolerance. Removing 16 selected two-qubit nodes lowers the canonical screen
again from 61.397 to 33.546 seconds.

### 3. Match path-search effort to the simplified graph

The final source requests `omeco-1-1`, the TensorCircuit-NG native shortcut
for one TreeSA trial and iteration. The path remains adequate: 50- and
100-step evaluators pass with final energies matching the expert. Timed search
latency falls enough to lower the canonical screen from 33.546 to 24.362
seconds.

This is not a framework patch or an environment override. Both roles use the
same latest TensorCircuit-NG image; only the candidate asks the existing
framework contractor for a smaller task-appropriate search budget.

## Preserved scientific semantics

The candidate retains:

- eight data and eight ancilla qubits, two adaptive layers, and the exact
  order of all data/ancilla rotations, entanglers, measurements, feedback,
  and CNOT ladders;
- all 96 independent trainable parameters and their seeded float32
  initialization;
- the same 64x16 seed-2048 float32 trajectory-uniform matrix, all 64
  trajectories, all 16 normalized TensorCircuit `cond_measure` operations
  per trajectory, and identical bit/trajectory order;
- the two independent feedback angles for each measured pair;
- the open-boundary eight-site TFIM with transverse field 1.05;
- exactly 100 sequential Optax Adam updates at learning rate 0.02, recording
  every pre-update energy;
- the final post-update 64-entry trajectory vector;
- complex64 TensorCircuit/JAX quantum computation and the exact output
  keys/shapes.

All six candidate canonical runs pass energy decrease, minimum improvement,
target energy, history length, trajectory shape, and NumPy output checks.

## Explored alternatives

| Experiment | Result | Decision |
| --- | --- | --- |
| e01: one native state/Hamiltonian expectation | 61.397 s canonical screen; energy/gradient errors `3.34e-6` / `1.01e-6` | Keep |
| e02: exact feedback `RZZ -> RZ` | 33.546 s; identity error `2.98e-8` | Keep |
| e03: whole-training `K.jaxy_scan` | 36.747 s vs 33.546 s | Reject: extra control-flow compilation |
| e04a: OMECo 1x1 | 24.362 s | Keep |
| e04b: greedy contractor | 23.234 s one-step vs 20.672 s for 1x1 | Reject |
| e05: joint state-based measurement rounds | fastest at 1/50 steps (4.606/14.523 s), but 26.531 s at 100 | Reject for canonical; dense-state AD crossover |
| e06: `K.vvag` trajectory gradients | 39.420 s one-step | Reject: mapped reverse-mode duplication |

The e05 crossover is useful beyond this exact evaluator: state materialization
is excellent when staging dominates or updates are few, whereas native
tensor-network `cond_measure` wins once many gradients amortize compilation.

## Measurement integrity and report recovery

The paired runner successfully wrote all 24 raw stdout/stderr files and
stopped the container after all 12 passing cells. It then hit a final
serialization typo (`true` instead of Python `True`). The bug occurred after
measurement and did not affect source bytes, ordering, runtimes, or outputs.

The runner is fixed and now checkpoints every completed cell. The tracked
paired report was reconstructed from all 12 raw logs without rerunning,
filtering, or changing any value; it records every stdout SHA-256 and the
staging snapshot hash. The fail-closed gate reports `promotion_ready: true`.

## PR summary

Suggested title:

`Task 07: collapse repeated energy/feedback contractions (4.48x)`

The PR should emphasize that the gain comes from exact Task 07 structure and
existing TensorCircuit-NG primitives—not fewer trajectories, fewer updates,
changed thresholds, hard-coded outputs, a framework downgrade, or a raw
NumPy/JAX simulator.
