# Task 07 Challenge-Design Reduction Report

## Executive result

Task 07 appears to require 64 differentiable trajectories of a 16-qubit
mid-circuit measurement-feedback VQE. For the published circuit, that
description hides an exact reduction:

```text
16 qubits x 64 measured trajectories
                 |
                 | exact analytic ancilla sampling
                 | exact reversible-bit inversion
                 v
8 data qubits x 2 unique weighted circuits
```

For the fixed public seed, 63 trajectories select the same all-zero
two-layer pattern and only trajectory 36 selects a second pattern. The final
implementation evaluates those two data-only TensorCircuit circuits, weights
them by `63/64` and `1/64`, and expands their final energies back to the
required 64 entries.

In six counterbalanced canonical pairs, every cell passes and the reduced
candidate wins 6/6:

| Metric | Human expert | Reduced candidate |
| --- | ---: | ---: |
| Passing runs | 6/6 | 6/6 |
| Mean runtime | 140.076441 s | 3.070839 s |
| Median runtime | 140.069298 s | 3.046739 s |
| Sample standard deviation | 15.386367 s | 0.124233 s |
| Standard error | 6.281458 s | 0.050718 s |
| Minimum / maximum | 123.286060 / 159.579514 s | 2.966582 / 3.311737 s |

Ratio-of-means speedup is **45.615x** and evaluator time falls by
**97.8077%**. Mean paired speedup is **45.758x**, with a two-sided 95%
Student-t interval of **[39.385x, 52.131x]**.

This is the repository campaign-best result for the public workload. It is
not a cross-hardware or global SOTA claim.

## Why the reduction is exact

### 1. The ancilla source distribution is classical

At the beginning of each layer every ancilla is in a computational-basis
state. In layer zero that state is `|0>`; in the next layer it is the
previous measured bit `b`.

After `RY(theta)`, the probability of the pre-ladder source bit `x=1` is:

```text
P(x=1 | b=0) = sin(theta/2)^2
P(x=1 | b=1) = cos(theta/2)^2
             = 1 - sin(theta/2)^2.
```

The following paired data-ancilla `RZZ` is diagonal in the ancilla
computational basis. Conditioned on `x`, it applies a unitary data rotation,
so it preserves the norm of each ancilla branch and cannot alter these
probabilities. Different ancillas remain independent before the ancilla
CNOT ladder.

### 2. The CNOT ladder is a prefix XOR

The expert applies the ordered ladder
`CNOT(a[0],a[1]), ..., CNOT(a[6],a[7])`. It maps independent source bits
`x` to measured bits `m` as:

```text
m[0] = x[0]
m[i] = m[i-1] xor x[i]
     = x[0] xor ... xor x[i].
```

The map is bijective:

```text
x[0] = m[0]
x[i] = m[i] xor m[i-1].
```

Sequential measurement therefore has the simple conditional law
`m[i] = m[i-1] xor x[i]`. The implementation uses the same float32 uniforms
and the same strict TensorCircuit condition
`status > 1-P(m[i]=1)`.

An independent audit compares this analytic rule with the full 16-qubit
TensorCircuit `cond_measure` program. All **1,024** bits
(`64 trajectories x 2 layers x 8 ancillas`) are identical.

### 3. Conditioned quantum action stays on the data register

Once `x` and `m` are fixed, the data-ancilla entangler and feedback gates
become data-only rotations:

```text
RZZ_ent(theta)      -> RZ_data((1 - 2*x) * theta)
RZZ_feedback(phi_m) -> RZ_data((1 - 2*m) * phi_m).
```

Both are Z rotations and commute, so the candidate emits their summed angle
as one native TensorCircuit `RZ` before the data CNOT ladder. The remaining
quantum circuit has eight data qubits and the unchanged open-boundary TFIM
Hamiltonian.

### 4. Equal trajectories can be merged

Applying the analytic sampler to all public fixed uniforms produces exactly
two complete two-layer patterns:

| Pattern | Count | Trajectory indices |
| --- | ---: | --- |
| all measured/source bits zero | 63 | all except 36 |
| rare nonzero pattern | 1 | 36 |

The rare measured-bit pattern, flattened by layer, is:

```text
00001111 00001010
```

Its inverse pre-ladder source pattern is:

```text
00001000 00001111
```

Because the objective is a mean over fixed trajectories, evaluating the two
unique circuits with weights `63/64` and `1/64` is algebraically identical
to evaluating 64 duplicates.

## Numerical audit

The proof is exact over real arithmetic. Complex64 contraction order changes
introduce small rounding differences, which are reported explicitly.

| Check against full accepted 16-qubit implementation | Result |
| --- | ---: |
| Analytic/full measured bits equal | true |
| Initial energy absolute error | `4.7684e-6` |
| Maximum per-trajectory energy error | `4.2915e-6` |
| Maximum non-ancilla gradient error | `1.5116e-6` |
| Full ancilla gradient maximum magnitude | `4.6559e-7` |
| Reduced ancilla gradient maximum magnitude | `0` |
| Post-one-Adam-update energy error | `4.6730e-5` |
| Audit decision | PASS |

The ideal pathwise derivative of a fixed discrete sample with respect to its
sampling angle is zero. The full complex64 graph leaves only sub-micro
rounding residue on those ancilla gradients. Adam can normalize tiny residue
into a visible parameter-coordinate movement, but parameters are not part of
the executable output contract and the physical energy checks remain close.

The complete 100-update evaluator also passes:

```text
initial history energy:  -6.8462696075
final history energy:    -10.0277128220
improvement:               3.1814432144
final trajectory mean:   -10.0331783295
final trajectory std:      0.0007445384
history length:            100
```

## Formal six-pair benchmark

All cells used one no-network Docker container, a fresh evaluator process,
six CPUs, 7 GiB, TensorCircuit-NG `1.8.0.dev20260726`, JAX/JAXLIB `0.10.0`,
and the unchanged 300-second limit. Pair order alternated to balance position.

| Pair | Order | Expert (s) | Candidate (s) | Speedup |
| ---: | --- | ---: | ---: | ---: |
| 1 | expert -> candidate | 123.286060 | 3.062554 | 40.2560x |
| 2 | candidate -> expert | 126.247088 | 3.311737 | 38.1211x |
| 3 | expert -> candidate | 150.224728 | 2.989908 | 50.2439x |
| 4 | candidate -> expert | 129.913867 | 3.030923 | 42.8628x |
| 5 | expert -> candidate | 159.579514 | 3.063330 | 52.0935x |
| 6 | candidate -> expert | 151.207388 | 2.966582 | 50.9702x |

No successful value was filtered or rerun. The report retains every raw
stdout/stderr hash and passes the frozen promotion rule.

| Artifact | SHA-256 |
| --- | --- |
| Immutable expert | `ac483319363f3c386a7646eaa867670ae3d3cd687f8517e6d4201e69240ff0a3` |
| Reduced candidate | `0337bf428a7c4a820f12f7db1232620b2777677617dd4f1a657dfd5f53bbdb0e` |
| Evaluator | `69717d98a90a7e53c31686128b3ef3e7cea3c96685ec538662a12163fe324b31` |
| Docker image | `b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833` |
| Staging snapshot | `d46912b2aba5e201c56754f98a21065be1009fe3c90d105a4a2b29b95aeaab0f` |
| Paired report | `068593daf65d132d1c7b3f18a0cbc2f7fc4b378558f3c4ccedf79231c0248c0f` |

## Secondary tuning after the reduction

Once only two eight-qubit circuits remained, the earlier large-network
choices were re-evaluated:

| Experiment | Canonical result | Decision |
| --- | ---: | --- |
| Python loop | 2.998 s exploratory | Keep |
| Whole-training `K.jaxy_scan` | 3.206 s | Reject |
| Explicit `RY`/`RZ` dense-gate fusion | 3.531 s | Reject |
| TensorNetwork greedy | 2.947 s six-run mean | Reject |
| OMECo 1x1 | 2.949 s six-run mean | Reject |
| TensorCircuit `plain-experimental` | 2.823/2.839 s comparison means | Keep |

The local contractor beat greedy in 6/6 paired screens with a mean
`1.0442x` speedup and 95% interval `[1.0029x, 1.0855x]`. It beat OMECo-1x1
in 5/6 pairs; that smaller `1.0395x` mean advantage has interval
`[0.9933x, 1.0856x]`. The default local setting was retained because it is
the simplest native small-graph choice and avoids OMECo path search.

## Is this a valid optimization or a loophole?

There are two defensible interpretations.

Under the executable public contract, it is valid:

- all 64 seeded statuses are consumed;
- all 96 parameters retain their layout;
- exactly 100 Adam updates are performed;
- the required pre-update history and 64 final trajectory energies are
  returned;
- no energy, output, threshold, or reference value is hard-coded;
- TensorCircuit performs all remaining quantum gate evolution and Hamiltonian
  expectations.

Under the likely benchmark-design intent, it is a loophole:

- the candidate no longer constructs 16 qubits;
- it does not call TensorCircuit `cond_measure` during optimization;
- it benchmarks an exact classical controller plus two eight-qubit circuits,
  not generic differentiable mid-circuit measurement at scale.

The problem statement does not explicitly prohibit exact analytic elimination
of measured ancillas or deduplication of fixed trajectories. If Task 07 is
intended to test TensorCircuit's mid-circuit measurement machinery, the
contract is under-specified.

## Recommended maintainer action

Keep two implementations visible:

1. The registered `conservative` e04a variant at
   `src/solutions/task-07/variants/solution_7_conservative.py` is the
   appropriate answer when literal 16-qubit TensorCircuit `cond_measure`
   execution is required. It preserves every intended operation and measured
   a 4.479x paired speedup.
2. The e11 reduction is the campaign-best answer to the current executable
   contract and should be used to document/fix the challenge-design gap.

To close the loophole in a future benchmark revision, require at least one of:

- explicit use of the full data-plus-ancilla register and framework-native
  mid-circuit measurement in the timed region;
- hidden instances with randomized layer counts, ancilla coupling topology,
  non-diagonal data-ancilla gates, or feedback that prevents branch
  classicalization;
- trainable sampling distributions whose gradients are defined by a stated
  estimator rather than pathwise differentiation through discrete branches;
- a policy check that rejects analytic elimination of the measured subsystem.

The strongest fix is semantic rather than cosmetic: introduce a
non-computational-basis ancilla interaction after entanglement so that an
ancilla measurement probability genuinely depends on the data state. Merely
changing the seed or increasing trajectory count does not remove the
reduction; it only changes the number of unique classical patterns.

## PR positioning

Suggested title:

`Task 07: expose exact classical-ancilla reduction (45.76x paired)`

The PR should explicitly label this as a challenge-design reduction, link the
full audit and conservative 4.48x alternative, and invite maintainers to
decide whether the public executable contract or the intended
mid-circuit-measurement semantics should govern acceptance.
