# Challenge 07: exact classical-ancilla reduction

## Summary

Challenge 07's reference solution is correct and faithfully implements the
stated 16-qubit measurement-feedback protocol. The public circuit itself,
however, has an unexpected exact reduction:

```text
16 qubits x 64 measured trajectories
                 |
                 | analytic ancilla sampling
                 | reversible prefix-XOR inversion
                 v
8 data qubits x unique weighted branches
```

For the published seed and configuration, the 64 complete two-layer
trajectories contain only two unique branches, with multiplicities 63 and 1.
The proof-of-concept implementation therefore evaluates two eight-qubit
TensorCircuit circuits, weights them by `63/64` and `1/64`, and expands their
final energies back to the required 64 entries.

This is best understood as a challenge-design loophole, not as a generic
acceleration of differentiable mid-circuit measurement. It satisfies the
public executable output contract while bypassing the intended 16-qubit
`cond_measure` workload.

The repository additions associated with this report do not replace the
canonical task or human reference:

- `optimized_sloutions/challenge-07/solution_7_classical_ancilla.py` is a
  runnable proof of concept.
- `scripts/audit_challenge_07_classical_reduction.py` independently compares
  the analytic reduction with a literal 16-qubit TensorCircuit circuit.

## Exact derivation

### Ancilla source probabilities are classical

At the start of each layer, every ancilla is in a computational-basis state.
For the first layer this is `|0>`; for the next layer it is the previous
measurement result `b`.

After `RY(theta)`, the pre-ladder source bit `x` has probability

```text
P(x=1 | b=0) = sin(theta/2)^2,
P(x=1 | b=1) = cos(theta/2)^2
             = 1 - sin(theta/2)^2.
```

The following data-ancilla `RZZ` is diagonal in the ancilla computational
basis. Conditioned on `x`, it applies a norm-preserving unitary to the data
qubit. It can change a branch phase and the data state, but cannot change the
ancilla Z-basis probability.

Consequently, the eight pre-ladder ancilla source bits are independent
Bernoulli variables even though the full state may be entangled with data.

### The ancilla CNOT ladder is a prefix XOR

Challenge 07 applies

```text
CNOT(a[0], a[1]), CNOT(a[1], a[2]), ..., CNOT(a[6], a[7])
```

in that order. On computational-basis source bits `x`, the measured bits `m`
are

```text
m[0] = x[0],
m[i] = m[i-1] xor x[i]
     = x[0] xor ... xor x[i].
```

This map is bijective:

```text
x[0] = m[0],
x[i] = m[i] xor m[i-1].
```

Sequential measurement can therefore be sampled with the same fixed
uniforms used by TensorCircuit. If `q_i = P(x[i]=1)`, then

```text
P(m[i]=1 | m[i-1]=0) = q_i,
P(m[i]=1 | m[i-1]=1) = 1 - q_i.
```

The proof of concept reproduces TensorCircuit's strict comparison rule
`status > 1-P(bit=1)`.

### Conditioned quantum action stays on the data register

Conditioning on one measured string selects one unique source string. The two
data-ancilla interactions become data-only Z rotations:

```text
RZZ_ent(theta)      -> RZ_data((1 - 2*x) * theta),
RZZ_feedback(phi_m) -> RZ_data((1 - 2*m) * phi_m).
```

They commute and can be emitted as one summed `RZ` angle. All remaining
quantum evolution is the original eight-data-qubit variational circuit and
the original open-boundary TFIM Hamiltonian.

### Equal fixed trajectories can be merged

For the public seed:

| Complete two-layer pattern | Count |
| --- | ---: |
| all measured and source bits zero | 63 |
| one rare nonzero pattern | 1 |

The rare measured-bit pattern, flattened by layer, is

```text
00001111 00001010
```

and its inverse pre-ladder source pattern is

```text
00001000 00001111.
```

The objective is an arithmetic mean over fixed trajectories. Evaluating each
unique branch once with its exact multiplicity is therefore algebraically
identical to evaluating all duplicates.

Changing only the seed does not close the fundamental loophole. It may
increase the number of unique branches, but every branch remains an
eight-qubit data-only circuit produced by an analytically sampled classical
ancilla controller.

### The branch table stays fixed during optimization

For a fixed sampled branch, the normalized conditioned data state does not
depend on the magnitude of its ancilla `RY` amplitude; that amplitude cancels
during projective normalization. The discrete comparison that selects the
branch has no pathwise derivative. Therefore the exact pathwise gradients of
all ancilla sampling angles are zero.

Adam leaves those angles unchanged, so the fixed-uniform branch table can be
computed once before the 100 updates. The full complex64 graph produces tiny
nonzero numerical residue on nominally zero coordinates; this finite-
precision artifact is measured below rather than treated as physical signal.

## Independent numerical audit

The included audit constructs both:

1. a literal 16-qubit TensorCircuit program with `cond_measure`; and
2. the independently derived analytic sampler and eight-qubit reduced
   energy.

The recorded audit on the public configuration found:

| Check | Result |
| --- | ---: |
| Full versus analytic measurement bits | all 1,024 equal |
| Unique complete patterns | 2 |
| Pattern counts | 63, 1 |
| Initial-energy absolute error | `1.1921e-5` |
| Maximum per-trajectory energy error | `1.1444e-5` |
| Maximum non-ancilla gradient error | `1.9896e-6` |
| Full ancilla-gradient maximum magnitude | `4.6529e-7` |
| Reduced ancilla-gradient maximum magnitude | `0` |
| Post-one-Adam-update energy error | `3.3379e-6` |

The ideal pathwise derivative of a fixed discrete branch with respect to its
sampling angle is exactly zero. The full complex64 contraction leaves only
sub-micro numerical residue on those coordinates. Adam can normalize tiny
residue into a visible parameter-coordinate change; parameters are not part
of the task output, and the physical energy comparisons remain close.

These values come from the self-contained script in the repository using the
same six-CPU/7-GiB, no-network TensorCircuit image as the canonical proof-of-
concept run. The separate portable benchmark audit used a different full-
circuit contraction order and reported comparably small errors; both records
pass their predeclared complex64 tolerances.

Run the audit from the repository root:

```bash
python3 scripts/audit_challenge_07_classical_reduction.py
```

Run the proof of concept with the canonical evaluator by copying it to a
temporary module or making the variant directory importable:

```bash
PYTHONPATH="$PWD/optimized_sloutions/challenge-07" \
  python3 tasks/challenge-07/tests/evaluate_7.py \
  --solution solution_7_classical_ancilla
```

## Paired performance evidence

The reduction was benchmarked in the separate portable expert benchmark
repository using the unchanged public evaluator, one no-network container,
six CPUs, 7 GiB, fresh evaluator processes, and alternating pair order.

| Metric | Human expert | Reduced candidate |
| --- | ---: | ---: |
| Passing runs | 6/6 | 6/6 |
| Mean runtime | 140.076441 s | 3.070839 s |
| Median runtime | 140.069298 s | 3.046739 s |
| Sample standard deviation | 15.386367 s | 0.124233 s |
| Standard error | 6.281458 s | 0.050718 s |

The candidate won 6/6 pairs. Ratio-of-means speedup was 45.615x, while mean
paired speedup was 45.758x with a two-sided 95% Student-t interval of
[39.385x, 52.131x]. No successful value was filtered or rerun.

Complete evidence, raw-output hashes, the conservative literal-measurement
variant, and the full derivation are available in
[OrbitBreakersExpertBenchmarks PR #11](https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks/pull/11).

Relevant immutable hashes:

| Artifact | SHA-256 |
| --- | --- |
| Human expert source | `ac483319363f3c386a7646eaa867670ae3d3cd687f8517e6d4201e69240ff0a3` |
| Reduced proof of concept | `0337bf428a7c4a820f12f7db1232620b2777677617dd4f1a657dfd5f53bbdb0e` |
| Evaluator | `69717d98a90a7e53c31686128b3ef3e7cea3c96685ec538662a12163fe324b31` |
| Six-pair report | `068593daf65d132d1c7b3f18a0cbc2f7fc4b378558f3c4ccedf79231c0248c0f` |

These timings are same-machine workload evidence, not a cross-hardware or
global SOTA claim.

## Why the current contract permits the workaround

The task prose says that Challenge 07 is designed to exercise mid-circuit
measurement and branch-dependent feedback. The executable checks only:

- history and trajectory array shapes;
- energy decrease and target thresholds;
- use of the selected quantum framework;
- absence of obvious hard-coded or raw-simulator bypasses.

The reduced solution still uses TensorCircuit for all quantum gates and
Hamiltonian expectations, consumes the fixed status matrix, trains the
original parameter layout, performs 100 Adam updates, and returns the required
64 trajectory energies. It does not need to construct the ancilla register or
call `cond_measure` during optimization.

This makes framework-fidelity review ambiguous: the solution is
mathematically faithful to the public instance but does not exercise the
framework feature that the task was intended to measure.

## Additional specification inconsistencies

The current problem statement contains two smaller issues:

1. It fixes `n_trajectories = 64` but writes the objective as
   `1/128 * sum(t=1..128) E_t`. The reference and evaluator use the mean over
   64 trajectories.
2. Two consecutive protocol sections are both titled
   "Data Post-Processing Layer".

These documentation issues do not cause the classicalization, but correcting
them would make a future revision less ambiguous.

## Recommended benchmark response

There are two reasonable policies.

### Accept exact reductions

Treat analytic elimination as scientific problem solving. Under this policy,
the proof of concept is a valid optimized artifact and Challenge 07 measures
whether an agent notices hidden circuit structure rather than only whether it
uses a mid-circuit API.

The task description should then disclose that equivalent analytic
reductions are allowed, and performance comparisons should identify this
algorithmic change explicitly.

### Require literal mid-circuit measurement

If the intended axis is framework support for hybrid measurement-feedback
programs, the stronger fix is to redesign the circuit rather than rely only on
a source-policy rule.

Useful changes include:

- apply a non-diagonal ancilla operation after the data-ancilla interaction,
  so measurement probabilities depend on the data state;
- replace the ancilla-only CNOT ladder with a transformation that is not a
  classical basis permutation;
- include multiple hidden configurations with varied layer counts, topology,
  and gates;
- define the desired sampling-gradient estimator explicitly;
- require framework-native mid-circuit measurement in the timed region as a
  secondary policy check.

Changing only the random seed, trajectory count, or OMECo budget does not
remove the exact classical controller.

## Proposed disposition

Keep the new proof of concept and audit as research artifacts, without
replacing the canonical reference in this pull request. Maintainers can then
choose whether a future Challenge 07 revision should:

- embrace the reduction as a valid expert insight;
- strengthen policy to require literal `cond_measure`; or
- redesign the circuit so the intended hybrid quantum-classical workload is
  intrinsic rather than merely described.
