# Task 01 MPO Energy Comparison

## Summary

This report records the evidence for promoting the Task 01 MPO energy variant
into `src/solutions/task-01/solution_1.py`.

The candidate replaces the reference solution's per-objective sum over 63
separate Pauli-pattern measurement contractions with a TensorCircuit-NG native
MPO Hamiltonian expectation. The variational ansatz, evaluator-provided DMRG
MPS input, Adam optimizer, learning rate, parameter initialization, seed
semantics, update count, and returned `energy_history` contract are preserved.

## Optimization Mechanism

The reference solution builds one MPS-backed circuit per objective and evaluates
31 two-site `ZZ` terms plus 32 one-site `X` terms through
`parameterized_measurements` and `K.vmap`.

The candidate builds the same TFIM Hamiltonian as an MPO and evaluates it with
`tensorcircuit.templates.measurements.mpo_expectation`, reducing the number of
separate observable contractions performed inside each objective.

`quimb` is used only to construct the TFIM MPO data structure, matching the
evaluator's Hamiltonian-construction style. The MPO is immediately converted
with `tc.quantum.quimb2qop`, and the circuit evolution plus energy expectation
remain TensorCircuit-NG native through `tc.Circuit` and `mpo_expectation`.

## Source Integrity

- Base repository commit: `3cd8c4733dbdf6752653c88ddb8b32df7d86bdbf`.
- Candidate solution commit: `4cac9cf`.
- Reference solution SHA256: `8B3F65B998DF1ECD2C4F9E66ACCE99CE2F5155E7A385678DE768FE9ABD6B4F02`.
- Candidate solution SHA256: `A6DCED0028B572EB36FB21A571D19F80DD2346C44417E7A17D1F668C883DE7C4`.
- Evaluator SHA256: `B05694154C1FE10C7EBCA4E153791B5AB3D29D9B76EE9725C00393CB44DD78B1`.
- Docker image used for exploratory timing: `orbitq-tensorcircuit-ng-1.8:py311`.
- Docker image ID: `sha256:8abafcb68d446f36665ca2cc27f5770e7f956a74aa579333afc739060d658d7e`.

## Code Change

The candidate makes one solution-file change:

- imports `quimb.tensor as qtn`;
- imports `mpo_expectation` instead of `parameterized_measurements`;
- replaces `tfim_measurement_data` with `tfim_mpo`;
- changes `circuit_energy` to call `mpo_expectation(circuit, mpo)`;
- constructs the MPO once in `run_solution(config)` and reuses it for all
  optimizer steps.

No evaluator, reference solution, task config, runner, or benchmark policy file
is changed.

## Correctness Evidence

Reduced comparison used a shared evaluator-generated DMRG MPS input,
`max_steps=20`, identical config, and identical seed/initialization.

- Returned keys: `["energy_history"]` for both reference and candidate.
- Shape: `(20,)` for both histories.
- NumPy dtype: `float32` for both histories.
- Max absolute history difference: `4.57763671875e-05`.
- Max relative history difference: `1.1029551661592766e-06`.
- `numpy.allclose`: `true` with `rtol=1e-5`, `atol=1e-5`.

Official Challenge 01 evaluator result for the candidate:

```text
End-to-end solution time: 64.78s
DMRG reference energy: -41.50398520
Initial variational energy: -41.50401306
Final variational energy: -41.50417328
Overall: PASS
```

Static policy result:

```text
static_policy_score: 1.0
line_count: 65
imports: numpy,optax,quimb,tensorcircuit
```

## Runtime Evidence

The exploratory timing campaign used fresh Docker processes, `--network none`,
the same fixed DMRG MPS input for reference and candidate, isolated cache/temp
directories per run, and NumPy materialization before stopping the timer.

Raw timings:

| Run | Pair order | Kind | Evaluator runtime (s) | External wall (s) |
|---:|---|---|---:|---:|
| 1 | reference -> candidate | reference | 105.580721535 | 109.807 |
| 2 | reference -> candidate | candidate | 67.021140770 | 71.462 |
| 3 | candidate -> reference | candidate | 65.715771778 | 70.038 |
| 4 | candidate -> reference | reference | 100.616901290 | 104.785 |
| 5 | reference -> candidate | reference | 102.546826054 | 106.777 |
| 6 | reference -> candidate | candidate | 67.121434447 | 71.379 |

Summary:

- Reference raw runtimes: `105.580721535`, `100.616901290`, `102.546826054`.
- Candidate raw runtimes: `67.021140770`, `65.715771778`, `67.121434447`.
- Median reference: `102.546826054`.
- Median candidate: `67.021140770`.
- Candidate/reference median ratio: `0.6535662131`.
- Speedup: `1.5300668547x`.
- Median runtime reduction: `34.6433786895%`.
- Paired direction: all three pairs favored the candidate.

## Interpretation

This is a solution-only TensorCircuit-NG optimization that preserves the Task 01
physics and output contract while reducing repeated observable contraction work.
The available paired measurements support submitting the MPO-energy candidate
for Task 01.

This report records exploratory evidence from the ORBIT-Q attempt workflow. A
larger `./bench run 01 --solution optimized --compare-to reference --repeat 6`
campaign can be added if maintainers want claim-quality OrbitBreakers-native
runtime statistics.
