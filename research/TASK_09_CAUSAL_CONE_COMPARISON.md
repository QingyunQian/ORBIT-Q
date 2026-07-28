# Task 09 causal-cone optimization comparison

Date: 2026-07-28

Base commit: `d612cd3ae752a8d16fd0b59c717d19abd4fb5f38`

## Result

The Task 09 candidate reduces evaluator-reported runtime from `67.624470` to
`18.204267` seconds in one matched Docker pair. Both processes passed the
functional evaluator. The measured speedup is `3.714759x`, or a `73.080355%`
runtime reduction.

| Measurement | Reference | Candidate |
| --- | ---: | ---: |
| Evaluator runtime | 67.624470 s | 18.204267 s |
| Observable history shape | `(200, 100)` | `(200, 100)` |
| Mean initial objective | -0.0022892463 | -0.0022892461 |
| Mean final objective | 1.5645751637 | 1.5645751691 |
| Final variance | 1.1838657734e-10 | 1.1816455867e-10 |
| Best final objective | 1.5645909309 | 1.5645909309 |
| Success fraction | 1.000000 | 1.000000 |
| Evaluator result | PASS | PASS |

One matched pair cannot establish a confidence interval. Treat the timing as
engineering evidence for the implementation, not as a promotion measurement.
The benchmark protocol requires at least six matched pairs for a reportable
runtime claim.

## Code comparison

The immutable
[reference](../references/task-09/solution_9.py#L45) constructs a 512-qubit
TensorCircuit circuit, applies 512 Hadamards and all 3,897 parameterized gates,
then asks `expectation_ps` to cancel the irrelevant circuit for each Pauli
term. The reference also keeps 3,897 parameters and two Adam moment arrays for
each of 200 restarts. A Python loop dispatches the jitted optimizer step 100
times.

The reference constructs the full circuit before TensorCircuit sees an
observable:

```python
# references/task-09/solution_9.py
circuit = tc.Circuit(config["n_qubits"])
for qubit in range(config["n_qubits"]):
    circuit.h(qubit)

for gate in gate_tape:
    if len(gate) == 3:
        getattr(circuit, gate[0])(gate[1], theta=params[gate[2]])
    else:
        getattr(circuit, gate[0])(
            gate[1], gate[2], theta=params[gate[3]]
        )
```

It then evaluates both terms from that 512-qubit circuit:

```python
for coeff, (xs, ys, zs) in pauli_data:
    total += coeff * K.real(
        circuit.expectation_ps(
            x=xs,
            y=ys,
            z=zs,
            enable_lightcone=True,
        )
    )
```

For an observable \(O\), TensorCircuit contracts the expectation network:

$$
\langle O \rangle = \langle 0 | U^\dagger O U | 0 \rangle.
$$

The network contains a ket copy of \(U\) and a conjugate bra copy of
\(U^\dagger\). With `enable_lightcone=True`, TensorCircuit cancels matching
gate pairs outside the backward influence of \(O\). Those pairs contribute
\(G^\dagger G = I\), so their removal preserves the expectation value. In the
reference, TensorCircuit performs this cancellation after Python has created
all 512 qubits and 3,897 gates.

The [candidate](../src/solutions/task-09/solution_9.py#L32) changes that path:

1. `extract_cone` scans the supplied gate tape backward from each measured
   Pauli support.
2. The scan retains a one-qubit gate when its qubit belongs to the current
   support. It retains a two-qubit gate when either endpoint belongs to the
   support, then adds both endpoints.
3. The candidate maps the retained qubits to compact indices and builds one
   TensorCircuit circuit per Pauli term.
4. The candidate gathers the union of active parameter indices after
   generating each full seeded initialization row.
5. `parameter_groups` separates terms whose active parameter sets do not
   intersect.
6. One jitted `jax.lax.scan` executes all 100 Adam updates for each parameter
   group.
7. TensorCircuit still applies `enable_lightcone=True` to each compact circuit.

The candidate performs the first pruning pass on the framework-neutral tape:

```python
# src/solutions/task-09/solution_9.py
support = {qubit for _, qubit in term}
retained = []

for gate in reversed(gate_tape):
    if len(gate) == 3:
        relevant = gate[1] in support
    elif len(gate) == 4:
        relevant = gate[1] in support or gate[2] in support
        if relevant:
            support.update((gate[1], gate[2]))
    else:
        raise ValueError(f"Invalid gate-tape entry: {gate}")
    if relevant:
        retained.append(gate)
```

It constructs a TensorCircuit circuit from the retained gates and asks
TensorCircuit to simplify the remaining expectation network:

```python
total = 0.0
for cone in cones:
    circuit = tc.Circuit(cone["n_qubits"])
    for qubit in range(cone["n_qubits"]):
        circuit.h(qubit)
    for gate in cone["gates"]:
        if len(gate) == 3:
            getattr(circuit, gate[0])(
                gate[1], theta=params[positions[gate[2]]]
            )
        else:
            getattr(circuit, gate[0])(
                gate[1], gate[2], theta=params[positions[gate[3]]]
            )
    xs, ys, zs = cone["paulis"]
    total += cone["coeff"] * K.real(
        circuit.expectation_ps(
            x=xs,
            y=ys,
            z=zs,
            enable_lightcone=True,
        )
    )
```

The two code paths use separate cancellation stages:

| Stage | Code owner | Removed work |
| --- | --- | --- |
| Backward gate-tape scan | Candidate | Gates and qubits outside the connectivity cone, before TensorCircuit graph construction |
| `enable_lightcone=True` | TensorCircuit | Redundant bra-ket structure in the compact expectation network, before contraction |

Removing `enable_lightcone=True` from the second code block produced the
300-second timeout recorded below. The explicit tape scan cuts graph
construction from 3,897 gates to 74 and 80 gates. TensorCircuit's cancellation
then makes the remaining doubled networks cheap enough for 200 restarts and
100 optimizer steps.

The public tape produces these structures:

| Observable | Reference circuit | Compact circuit | Retained gates | Active parameters |
| --- | ---: | ---: | ---: | ---: |
| `X_388 Z_390` | 512 qubits | 18 qubits | 74 | 74 |
| `X_16 Y_19` | 512 qubits | 15 qubits | 80 | 80 |

The parameter sets do not overlap. The candidate therefore trains two
independent 74- and 80-coordinate problems and sums their pre-update objective
histories.

## Why the transformation preserves the task

Unitaries outside a local observable's backward causal cone cancel between the
ket and bra networks. Removing those gates before TensorCircuit graph
construction leaves the observable and its active gradients unchanged.

The initialization path generates the same 3,897-element NumPy row as the
reference for every restart and gathers the active entries afterward. Drawing
154 values would assign different random values to the scattered active
indices, so the candidate does not use that shortcut.

Parameters outside both cones have zero gradients. Their Adam moments remain
zero and their values do not change. Adam updates each coordinate from its own
gradient and moment state, so parameter-disjoint loss terms can run in
separate scans. The candidate combines terms into one group when their
parameter sets overlap.

`jax.lax.scan` preserves the reference update order. Each history entry still
records the objective before its corresponding Adam update. The candidate
returns the required `(n_restarts, max_steps)` NumPy array.

The packed parameter, first-moment, and second-moment arrays contain 25.3 times
fewer coordinates:

```text
3897 / 154 = 25.305
```

For 200 restarts in float32, those three persistent arrays shrink from about
8.92 MiB to 0.35 MiB. This figure excludes compiled intermediates and
TensorCircuit contraction storage.

## Effort record

### 1. Structural inspection

We inspected the evaluator, gate-tape generator, reference solution, output
contract, and optimizer schedule. A reverse connectivity scan reproduced the
reported 18- and 15-qubit cone sizes. It retained 74 and 80 gates, with no
shared parameter indices.

### 2. First compact implementation

The first candidate implemented backward-cone extraction, compact qubit
mapping, active-coordinate gathering, parameter-overlap grouping, and a
whole-training `jax.lax.scan`. It disabled TensorCircuit's automatic
light-cone cancellation because the input graph had already been pruned.

A two-restart, two-step comparison completed:

| Measurement | Reference | First candidate |
| --- | ---: | ---: |
| Runtime | 47.230586 s | 16.056370 s |
| Maximum history difference |  | 7.450581e-7 |
| `rtol=1e-6`, `atol=1e-6` |  | PASS |

We found a performance regression in the full public run. The reference passed
in `98.805672` seconds, while the candidate hit the 300-second timeout. The
failed candidate had source SHA-256
`2a49a983a2370e79640107d2c92b42628d4de1ad4be6c96ef84b5003c6058ade`.

### 3. Contraction-path correction

The compact circuit still creates a doubled tensor network for expectation
evaluation. Graph-level connectivity pruning did not remove enough work from
that network at 200-way vectorization. We restored
`enable_lightcone=True` on each compact circuit.

The corrected two-restart, two-step history matched the saved reference values
with a maximum absolute difference of `3.727175e-9`. The corrected candidate
completed that smoke run in `17.834426` seconds.

### 4. Full validation

A standalone full candidate run passed in `14.820049` seconds. We then ran one
fresh reference-to-candidate pair in the same staged task container:

```bash
./bench run 09 \
  --solution optimized \
  --compare-to reference \
  --repeat 1 \
  --engine docker \
  --timeout 300 \
  --no-build
```

The reference passed in `67.624470` seconds. The candidate passed in
`18.204267` seconds. The benchmark reported `3.714759x` paired speedup and
`73.080355%` improvement.

Static validation also passed:

```bash
python3 -m py_compile src/solutions/task-09/solution_9.py
git diff --check
./bench verify
```

The candidate contains 156 non-empty, non-comment lines, below the task's
200-line policy limit.

## Main findings

### Prune before graph construction

TensorCircuit's automatic cancellation reaches the relevant contraction after
the reference has created thousands of irrelevant gates and their tensor
nodes. A backward gate-tape scan removes that construction cost and gives
TensorCircuit two small circuits.

### Keep TensorCircuit cancellation after pruning

The 300-second timeout provides the strongest ablation in this comparison.
Explicit connectivity pruning and TensorCircuit light-cone cancellation solve
different parts of the cost. The first limits the circuit that enters the
framework. The second simplifies the doubled expectation network inside that
compact circuit. The final candidate needs both.

### Pack inactive optimizer coordinates

Only 154 of 3,897 parameters affect the requested observables. Gathering those
coordinates cuts persistent Adam storage and elementwise optimizer work by
25.3 times while preserving the seeded initial values.

### Exploit separability when the tape permits it

The two public cones share no trainable parameters. Coordinatewise Adam allows
the candidate to train them in separate scans. The code detects overlap from
the supplied tape and combines intersecting terms, so it does not hardcode the
public qubit positions, cone sizes, or separation result.

### Compile the optimizer trajectory as one unit

The reference invokes one jitted step from Python 100 times. The candidate
places the complete Adam trajectory inside one jitted scan and returns the
stacked pre-update history.

## Attribution limits and next checks

The final timing measures cone extraction, coordinate packing, separation, and
`lax.scan` together. We did not measure independent full-size ablations for
packing, grouping, or scan, so the evidence does not assign a speedup to any
one of those changes.

The comparison covers the deterministic public Task 09 configuration on one
host and one pinned image. A follow-up performance report should run at least
six counterbalanced pairs and report the mean, median, standard error, pair
wins, and paired-speedup confidence interval. Separate ablations could then
measure:

- compact cones with full optimizer coordinates;
- packed coordinates with a Python optimizer loop;
- one combined 154-coordinate loss instead of two disjoint scans.

## Provenance

| Artifact | SHA-256 or version |
| --- | --- |
| Reference solution | `b28a9df18a46cb2e211a02bf526f3c3b75a44e11e7d364ec5f81026202d8d1d9` |
| Final candidate solution | `72f30ded4c8695b2dbbd0d3b3054b3ce349a8cc9406d6864e98f38fee46462f8` |
| Task 09 evaluator | `9ae54eed501fa71d985ab92aa3601714f41cf9848bea0014c0d8cfb7c866f58d` |
| Requirements lock | `cd5ac5cb2102ea7b40bd46dc81320cc59e0ce0671ab88c597f81d82b384a824b` |
| Docker image ID | `sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c` |
| TensorCircuit-NG | `1.7.0.dev20260618` |
| JAX/JAXLIB | `0.10.0` |
| CPU and memory limits | 8 CPUs, 9 GiB |
| Evaluator timeout | 300 seconds |
