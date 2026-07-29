# Challenge 07: exact classical-ancilla reduction

## Finding

Challenge 07's reference is correct, but the published circuit admits an
unexpected exact reduction:

```text
16 qubits x 64 measured trajectories
                 |
                 | analytic ancilla sampling
                 | prefix-XOR inversion
                 v
8 data qubits x unique weighted branches
```

For the public seed, the 64 complete two-layer trajectories contain only two
unique branches with multiplicities 63 and 1. The proof of concept evaluates
those two eight-qubit TensorCircuit circuits, weights them by `63/64` and
`1/64`, and reconstructs the required 64 final energies.

This is a challenge-design loophole, not a generic acceleration of
framework-native mid-circuit measurement. The canonical task, evaluator, and
human reference are unchanged by this variant.

Files in this directory:

- `solution_7_classical_ancilla.py`: runnable proof of concept;
- `audit_classical_ancilla_reduction.py`: literal 16-qubit versus reduced
  numerical audit.

## Why the reduction is exact

At the start of each layer, every ancilla is in a computational-basis state
`|b>`. After `RY(theta)`, its pre-ladder source bit `x` satisfies

```text
P(x=1 | b=0) = sin(theta/2)^2,
P(x=1 | b=1) = cos(theta/2)^2.
```

The following data-ancilla `RZZ` is diagonal in the ancilla basis. Conditioned
on `x`, it applies a norm-preserving data unitary, so it cannot change the
ancilla Z-basis probability.

The ordered ancilla ladder

```text
CNOT(a[0], a[1]), ..., CNOT(a[6], a[7])
```

is the reversible prefix-XOR map

```text
m[0] = x[0],
m[i] = m[i-1] xor x[i],
x[i] = m[i] xor m[i-1].
```

It can therefore be sampled analytically with the same fixed uniforms and
TensorCircuit's strict `status > 1-P(bit=1)` comparison.

Conditioning on source and measured bits converts the two data-ancilla
interactions into data-only rotations:

```text
RZZ_ent(theta)      -> RZ_data((1 - 2*x) * theta),
RZZ_feedback(phi_m) -> RZ_data((1 - 2*m) * phi_m).
```

The rotations commute and are emitted as one summed `RZ` angle. The remaining
quantum computation is the original eight-data-qubit circuit and TFIM
expectation.

For a fixed branch, projective normalization cancels the magnitude of the
ancilla `RY` amplitude. The discrete comparison has no pathwise derivative,
so the exact ancilla-angle gradients are zero and the branch table remains
fixed during Adam optimization.

## Reproduction

Run the self-contained audit from the repository root:

```bash
python3 optimized_solutions/challenge-07/audit_classical_ancilla_reduction.py
```

Run the 100-step proof of concept:

```bash
PYTHONPATH="$PWD/optimized_solutions/challenge-07" \
  python3 tasks/challenge-07/tests/evaluate_7.py \
  --solution solution_7_classical_ancilla
```

The no-network, six-CPU/7-GiB TensorCircuit run produced:

| Check | Result |
| --- | ---: |
| Canonical evaluator | PASS, 3.15 s |
| Full versus analytic measurement bits | all 1,024 equal |
| Unique branch counts | 63, 1 |
| Initial-energy absolute error | `1.1921e-5` |
| Maximum trajectory-energy error | `1.1444e-5` |
| Maximum non-ancilla gradient error | `1.9896e-6` |
| Full / reduced ancilla-gradient max | `4.6529e-7` / `0` |
| Post-one-Adam-update energy error | `3.3379e-6` |

The tiny full-circuit ancilla gradient is complex64 contraction residue on an
exactly zero pathwise derivative.

## Paired performance evidence

A separate same-container benchmark used six CPUs, 7 GiB, no network, fresh
evaluator processes, and alternating pair order:

| Metric | Human expert | Reduced variant |
| --- | ---: | ---: |
| Passing runs | 6/6 | 6/6 |
| Mean runtime | 140.076441 s | 3.070839 s |
| Median runtime | 140.069298 s | 3.046739 s |

The reduced variant won 6/6 pairs. Mean paired speedup was 45.758x with a 95%
Student-t interval of [39.385x, 52.131x]. No successful value was filtered or
rerun. Full logs and hashes are preserved in
[OrbitBreakersExpertBenchmarks PR #11](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/11).

The proof-of-concept SHA-256 is
`0337bf428a7c4a820f12f7db1232620b2777677617dd4f1a657dfd5f53bbdb0e`.
These are same-machine workload results, not a cross-hardware SOTA claim.

## Benchmark implications

The executable checks allow this solution because it still:

- uses TensorCircuit for all remaining quantum evolution and expectations;
- consumes all fixed status rows and keeps the 96-parameter layout;
- performs exactly 100 Adam updates;
- returns the required history and 64 trajectory energies.

It does not construct the ancilla register or call `cond_measure` during
optimization. If Challenge 07 is intended to measure framework support for
hybrid measurement-feedback programs, the stronger fix is to redesign the
circuit so that measurement probabilities depend on the data state, for
example with a non-diagonal ancilla operation after data-ancilla interaction.
Changing only the seed or trajectory count does not remove the analytic
controller.

The problem statement also defines 64 trajectories but writes a `1/128`
objective, and repeats the "Data Post-Processing Layer" heading. These
documentation issues are independent of the reduction.

Maintainers can either accept the reduction as scientific problem solving or
revise the circuit/policy to require intrinsic framework-native mid-circuit
measurement.
