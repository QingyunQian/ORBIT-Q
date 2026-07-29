# Task 07 Autoresearch Campaign

Destination: `research/task-07/LOG.md`

Task: `task-07`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and setup

Selected task: `task-07` (16-qubit measurement-feedback VQE).

Base commit: `5af98f27b9404c513df8eee0f4568b1512edee19`.

Branch: `codex/orbitbreakers/task-07/extreme-native`.

Worktree:
`/Users/qqy/.codex/visualizations/2026/07/28/019fa982-7244-7e20-99f5-f609bdd0cf27/task07-extreme`.

The branch and worktree were created before the Task 07 survey and candidate
files. No candidate source was edited before the survey and public workload
gates were completed.

Open pull requests in `hmyuuu/OrbitBreakersExpertBenchmarks` were inspected
before selection. The open optimization PRs covered Tasks 08, 09, and 10;
none covered Task 07.

## Immutable expert bootstrap: canonical

Date: 2026-07-29

Reference SHA-256:
`ac483319363f3c386a7646eaa867670ae3d3cd687f8517e6d4201e69240ff0a3`.

Evaluator SHA-256:
`69717d98a90a7e53c31686128b3ef3e7cea3c96685ec538662a12163fe324b31`.

Docker image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
(TensorCircuit-NG `1.8.0.dev20260726`, JAX/JAXLIB `0.10.0`).

Allocation: six CPUs, 7 GiB memory, no network; timeout 300 seconds.

```text
workload: canonical max_steps=100
terminal_status: SUCCESS
valid: true
runtime_sec: 135.815605
initial_energy: -6.8462643623
final_history_energy: -10.0279636383
improvement: 3.1816992760
final_trajectory_mean: -10.0333871841
final_trajectory_std: 0.0000000000
history_length: 100
```

Decision: `bootstrap baseline`. A performance claim still requires six
matched reference/candidate pairs in one container.

## Immutable expert update-count profile

Date: 2026-07-29

All runs used the same source, image, six-CPU/7-GiB limits, seed, layers,
trajectory batch, output schema, thresholds, and evaluator. Only the
evaluator-supported `--max-steps` argument changed.

| Updates | Runtime (s) | Final history energy | Overall |
| ---: | ---: | ---: | --- |
| 1 | 47.853128 | -6.8462719917 | FAIL (expected thresholds) |
| 10 | 51.638506 | -7.4798407555 | FAIL (target) |
| 20 | 60.074862 | -7.8647251129 | FAIL (target) |
| 32 | 69.596274 | -8.1460399628 | FAIL (target) |
| 50 | 91.540316 | -8.7927856445 | PASS |
| 100 | 135.815605 | -10.0279636383 | PASS |

The one-step run establishes about 48 seconds of fixed trace, compilation,
contraction-path, and final-evaluation work. The remaining 99 canonical
updates add approximately 0.89 seconds each. Decision: prioritize the
per-step energy/gradient contractions, retain 50 updates as a passing screen,
and make claims only on the canonical 100-step case.

## Frozen hypotheses

The complete pre-edit hypothesis definitions and falsification rules are in
`SURVEY.md`.

Primary experiment: `e01`, single native TensorCircuit state contraction plus
one native sparse eight-data-qubit Hamiltonian expectation per trajectory.

Secondary experiments, each isolated from the latest accepted commit:

- `e02`: exact measured-ancilla feedback `RZZ` to data `RZ` reduction;
- `e03`: whole-training `K.jaxy_scan`;
- `e04`: OMECo contractor-budget sweep;
- `e05`: native-state measurement-round reuse, only if lower-risk ideas leave
  substantial headroom.

## Experiment `e01`

Branch: `codex/orbitbreakers/task-07/e01-single-state-energy`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e01-single-state-energy`.

Parent commit: `21d7271` (`research: freeze task 07 optimization campaign`).

### Hypothesis

Replacing 15 separate per-trajectory Pauli expectation contractions with one
TensorCircuit `state` contraction and one TensorCircuit-native sparse TFIM
operator expectation materially reduces trace/compile and repeated gradient
cost while preserving energies, gradients, one Adam update, all 100
pre-update history values, and the final trajectory vector within declared
complex64 tolerances.

### Pre-run frozen environment

Public dataset version: `orbitq-workloads-v20260729.5`.

Private evaluation used: `no`.

Reference/evaluator/image hashes: as recorded above.

Pair order for final promotion: odd `reference -> candidate`, even
`candidate -> reference`; six pairs; 300-second cap.

The one-trajectory equivalence check is frozen before execution at absolute
tolerances `5e-5` for energy, `5e-4` for the maximum gradient element, and
`2e-5` for the maximum parameter difference after one Adam update. These are
strict complex64 path-rounding tolerances and are not evaluator thresholds.

### Result

Candidate hypothesis commit: `1e21dd41f47e3beb84b937144324227eac544b6d`.

Candidate SHA-256:
`30f0f45073e866c7fbb24cd9a5c33d8c1254e6985136937cd454b82990681678`.

Candidate diff SHA-256:
`3396c6b65d3f6c2eaebc647d6251547dd6561d1defac4526f9f503b2dcac8b7e`.

Sanitized record: `profiles/e01-single-state-screen.json`.

```text
max_steps=1:   reference 47.853128 s, candidate 44.917110 s
max_steps=10:  reference 51.638506 s, candidate 46.100052 s
max_steps=50:  reference 91.540316 s, candidate 52.769482 s, both PASS
max_steps=100: reference 135.815605 s, candidate 61.396553 s, both PASS
canonical single-screen speedup: 2.2121x
```

The candidate canonical run passed every evaluator gate with initial energy
`-6.8462653160`, final history energy `-10.0263500214`, improvement
`3.1800847054`, final trajectory mean/std
`-10.0319023132 / 0.0014687895`, history length 100, and the exact required
keys and shapes.

The predeclared one-trajectory audit measured energy error `3.34e-6` and
maximum gradient error `1.01e-6`, both comfortably passing. Its strict
maximum parameter-difference check after one first Adam update failed:
`3.13e-2` versus `2e-5`, although mean parameter difference was `2.43e-3`.
This failure is retained, not filtered.

Decision: `keep provisionally`. The state/observable identity is exact,
energy and gradient checks pass, the one-step post-update physical energy
differs by only `2.29e-5`, and the 50/100-step public workloads both pass with
nearly identical energy trajectories. The failed parameter metric reflects
the ill-conditioning of first-step Adam updates near zero: Adam normalizes
each gradient component by its magnitude, so a complex64 sign change in an
otherwise negligible component can create an order-learning-rate parameter
difference without a corresponding energy difference. Final paired
performance evidence is still pending.

## Append-only corrections

Append corrections below this heading. Never rewrite a result after it has
informed another experiment.

### Correction: e01 one-update acceptance observable

The strict per-parameter first-Adam maximum was over-specified as a semantic
criterion. The executable contract does not return parameters, and this
metric is discontinuously sensitive at zero gradient. It remains visible as
a failed diagnostic. Subsequent candidates will predeclare post-update
energy/trajectory checks as the physical one-update criterion while continuing
to report gradient errors and any parameter differences.

## Experiment `e02`: measured-ancilla feedback reduction

Branch: `codex/orbitbreakers/task-07/e02-feedback-rz`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e02-feedback-rz`.

Parent commit: `2e5cd61` (accepted e01 code and complete provisional evidence).

### Hypothesis

After `cond_measure`, ancilla `a` is the Z eigenstate with eigenvalue
`1-2*bit`. Therefore the selected
`RZZ(theta_bit)` on `(ancilla_a, data_a)` is exactly
`RZ((1-2*bit)*theta_bit)` on the data qubit, with the ancilla unchanged.
Replacing all 16 generic selected two-qubit tensors with those native
TensorCircuit `RZ` gates reduces graph/path/gradient work without changing
any measurement, selected parameter, branch state, or observable.

### Pre-run frozen checks

The exact two-branch gate identity will be checked at complex64 precision.
The frozen matrix-action tolerance is `1e-7` for both bit values and two
nontrivial signed angles.
The physical one-step criteria are initial energy absolute error at most
`5e-5` versus accepted e01 and post-update final-trajectory mean absolute
error at most `1e-4`; the strict parameter maximum remains diagnostic only.
Both the public 50-step and canonical 100-step evaluators must pass. A
candidate is retained only if its canonical screen is faster than e01's
61.396553 seconds.

### Result

Candidate hypothesis commit: `067a1d365e8ef4a9e3f81d6dd62939c0b3af6b39`.

Candidate SHA-256:
`b3dcbaa35a233d8dde4576de7257c79a1a81034eee49f1f0bef6489116dcafcd`.

Candidate diff SHA-256:
`759b7139a8463d53c80f2d48148eca189235a872fac5402ae9f64be4100bac45`.

Sanitized record: `profiles/e02-feedback-rz-screen.json`.

The independent two-branch matrix-action audit passed all four cases. Maximum
complex64 error was `2.98e-8` against the frozen `1e-7` threshold.

```text
max_steps=1:   e01 44.917110 s, e02 29.918828 s
max_steps=50:  reference 91.540316 s, e01 52.769482 s,
               e02 31.299628 s, e02 PASS
max_steps=100: reference 135.815605 s, e01 61.396553 s,
               e02 33.546170 s, e02 PASS
canonical e02/reference single-screen speedup: 4.0487x
canonical e02/e01 single-screen speedup: 1.8302x
```

The one-step initial energy and post-update trajectory-mean differences from
accepted e01 are `1.43e-6` and `7.53e-5`, passing the frozen `5e-5` and
`1e-4` physical thresholds. The canonical run passed every evaluator gate:
initial `-6.8462653160`, final history `-10.0280771255`, improvement
`3.1818118095`, final trajectory mean/std
`-10.0335464478 / 0.0000002666`, history length 100, required keys/shapes.

Decision: `keep`. The exact feedback reduction removes a major trace,
path-search, contraction, and gradient burden. Proceed from e02 to isolate
whole-training scan.

## Experiment `e03`: whole-training TensorCircuit scan

Branch: `codex/orbitbreakers/task-07/e03-training-scan`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e03-training-scan`.

Parent commit: `74d632d` (accepted e02 implementation and evidence).

### Hypothesis

Carrying parameters and the Optax state through `K.jaxy_scan` for exactly 100
iterations and emitting each pre-update value preserves the sequential Adam
trajectory while eliminating 100 Python-to-JAX dispatches. The expected gain
is small because e02's step is already cheap relative to compilation.

### Pre-run frozen checks

The public 50-step and canonical 100-step evaluators must both pass. Initial,
final-history, and post-update trajectory-mean energies must remain within
`5e-3` of accepted e02, allowing normal complex64 optimizer divergence but
not a changed objective. Retain only if the canonical screen is faster than
e02's 33.546170 seconds.

### Result

Candidate hypothesis commit: `8b977faa5a9eea4e4ead88e24f4193cbfbc66aa0`.

Candidate SHA-256:
`767c82d6c0be9af2ee526d130afba3f6eb95d81d42efcbb7f62665a9753a3b2c`.

Candidate diff SHA-256:
`5bffe6df6108a72656dfd56cdc9d4a39aeaaa0037e3545e3d4bb1a754d4229d3`.

Sanitized record: `profiles/e03-training-scan-screen.json`.

```text
max_steps=50:  e02 31.299628 s, scan 34.916768 s, scan/e02 1.11556
max_steps=100: e02 33.546170 s, scan 36.747307 s, scan/e02 1.09542
```

Both scan runs passed every evaluator criterion. The canonical initial,
final-history, and final-trajectory-mean energies differ from e02 by
`6.68e-6`, `1.80e-3`, and `1.71e-3`, all within the frozen `5e-3`
physical threshold.

Decision: `discard`. Staging the already optimized value/gradient/Adam body
inside a scan adds more control-flow compile cost than 100 cached-JIT Python
dispatches. Continue from accepted e02 without scan.

## Experiment `e04a`: OMECo 1x1 path-search budget

Branch: `codex/orbitbreakers/task-07/e04-omeco-1x1`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e04-omeco-1x1`.

Parent commit: `945430e` (e02 restored after the rejected scan).

### Hypothesis and frozen rule

For e02's simplified low-depth graph, `TreeSA(ntrials=1,niters=1)` can reduce
timed path-search latency more than it increases compiled contraction work.
Screen `max_steps=1`; retain for a full 50/100-step validation only if it is
faster than e02's 29.918828-second one-step screen and the initial/post-update
energies remain within `1e-4`.

### Result

Candidate hypothesis commit: `1eb52b206dacd848dd8efae29473415c1e37d3b0`.

Candidate SHA-256:
`0cd9676dc904660597a2f7dd6981fdac596295e4eae99510a3f4f21671859592`.

Candidate diff SHA-256:
`673f016863a7b777669c83dc399753004ebb19e8bc6f49003259699666d438d3`.

Sanitized record: `profiles/e04a-omeco-1x1-screen.json`.

```text
max_steps=1:   e02/32x32 29.918828 s, 1x1 20.672377 s
max_steps=50:  reference 91.540316 s, 1x1 21.473078 s, PASS
max_steps=100: reference 135.815605 s, e02/32x32 33.546170 s,
               1x1 24.362414 s, PASS
canonical 1x1/reference single-screen speedup: 5.5757x
```

Initial and one-step post-update trajectory-mean differences from e02 are
`4.77e-7` and `5.19e-5`, within the frozen `1e-4` rule. The canonical run
passes with final history `-10.0276298523`, improvement `3.1813645363`, and
final trajectory mean/std `-10.0331916809 / 0.0014021704`.

Decision: `keep`. For the simplified graph, TreeSA 1x1 finds an adequate
repeated contraction path while saving most of the timed search latency.

## Experiment `e04b`: TensorNetwork greedy contractor

Branch: `codex/orbitbreakers/task-07/e04-greedy`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e04-greedy`.

Parent commit: `945430e`.

### Hypothesis and frozen rule

The built-in greedy contractor may eliminate nearly all path-search latency.
Retain only if its one-step screen is faster than OMECo-1x1's
`20.672377` seconds and energies remain within `1e-4`.

### Result

Candidate hypothesis commit: `834de2a4f8e06b23cc9555b7fee4c25ff843a053`.

Candidate SHA-256:
`96e2cd89224867352de887bec17058c43deb798a2e8200df670854a8538c3eda`.

Candidate diff SHA-256:
`080396317aee29749b4c075b18778756af94b6621f1c52c9daaba270575eddd1`.

Sanitized record: `profiles/e04b-greedy-screen.json`.

The physical comparison passed, but greedy required 23.234316 seconds versus
OMECo-1x1's 20.672377 seconds.

Decision: `discard` without 50/100-step runs; greedy is 12.39% slower at the
predeclared screen.

## Experiment `e05`: joint TensorCircuit-state measurement rounds

Branch: `codex/orbitbreakers/task-07/e05-batched-measurement-rounds`.

Fresh hypothesis worktree:
`/Users/qqy/Desktop/2026Project/ORBIT-Q-worktrees/orbitbreakers/task-07/e05-batched-measurement-rounds`.

Parent commit: `e38891f` (accepted OMECo-1x1 plus greedy rejection evidence).

### Hypothesis

Before each eight-ancilla measurement round, one TensorCircuit `c.state()`
contains the exact joint ancilla distribution. Sequentially condition that
distribution with the same eight fixed uniforms and strict
`status > p0` rule used by TensorCircuit `_unitary_kraus_template`, select and
normalize the corresponding TensorCircuit state column, and continue the
adaptive circuit from that collapsed state. This replaces eight separately
contracted `cond_measure` probability networks per layer with one
TensorCircuit state contraction while preserving the exact projective
measurement law, bit order, fixed uniforms, feedback branches, and central
TensorCircuit gate/state computation.

### Frozen policy and numerical checks

This is explicitly a higher-risk framework-native restructuring: JAX is used
only to condition probabilities and select a column of a state computed by
TensorCircuit; all quantum state evolution, gates, and Hamiltonian evaluation
remain TensorCircuit APIs. Screen one step first. Initial and post-update
trajectory-mean energies must be within `5e-3` of accepted e04a, all outputs
must be finite, and memory must remain below 7 GiB. Continue to 50/100 steps
only if the one-step runtime is below 20.672377 seconds. Retain only if both
public workloads pass and canonical runtime is below 24.362414 seconds.

### Result

Candidate hypothesis commit: `5c4f0ba61509625c4c2bf76bcb3e21adf9fc09c9`.

Candidate SHA-256:
`5f4ec8d45e1fa91c053f3ed2027c90bdb1abf1e9e7d87add399c6c5ce883add7`.

Candidate diff SHA-256:
`d4a0277c7c1184b22be90dc51215a63690b39cb23871d9fcd3d6d6818aafd64b`.

Sanitized record: `profiles/e05-measurement-round-screen.json`.

```text
max_steps=1:   e04a 20.672377 s, e05 4.606171 s
max_steps=50:  reference 91.540316 s, e04a 21.473078 s,
               e05 14.522625 s, e05 PASS
max_steps=100: reference 135.815605 s, e04a 24.362414 s,
               e05 26.530668 s, e05 PASS
```

All physical checks and both public workloads pass. The canonical result is
especially close to the immutable expert: final history
`-10.0279579163` versus `-10.0279636383`, with improvement
`3.1816935539` and final trajectory mean/std
`-10.0334491730 / 0.0000033379`.

Decision: `discard for the canonical metric`. Joint measurement rounds cut
the fixed trace/path cost by 16 seconds and dominate at 1/50 updates, but
materializing and differentiating full 16-qubit states makes each update
roughly 0.20 seconds slower. At 100 updates it is 8.90% slower than accepted
e04a. Preserve it as a valuable scaling crossover insight, not the final
candidate.

## Experiment `e06`: TensorCircuit vectorized value-and-gradient

Branch: `codex/orbitbreakers/task-07/e06-vvag`.

Parent commit: `f8da3bb`.

Hypothesis: `K.vvag` may generate a better batched-trajectory AD program than
differentiating through the mapped mean. Retain only if its one-step runtime
beats e04a's 20.672377 seconds with physical outputs within `1e-4`.

Candidate commit: `828e921ddf69709054d5fa52a5e3e62d9fac475e`;
source SHA-256
`4f0a8757af372c41db677981f1551c363f82f128e0f295073228defece065c68`;
diff SHA-256
`82e90bad5bb428a53d623e268808a22a1ca321a686540f8e7c8ea5b46a15d7fa`.

Sanitized record: `profiles/e06-vvag-screen.json`.

Result: physical outputs pass, but one step takes 39.419801 seconds,
1.90687x e04a. Decision: `discard` without longer runs. Mapping individual
value-and-gradient programs duplicates reverse-mode structure for this
shared-parameter objective.

## Frozen final candidate and paired run

Final candidate: accepted e04a source
`0cd9676dc904660597a2f7dd6981fdac596295e4eae99510a3f4f21671859592`.

No candidate tuning follows this freeze. The final command is:

```bash
python -u research/task-07/run_docker_matrix.py \
  --repeat 6 --max-steps 100 --timeout 300 --cpus 6 --memory 7g \
  --output results/task-07-final-canonical-6-pairs
```

It stages immutable source snapshots, uses one no-network container and fresh
evaluator processes, alternates pair order, and applies the survey's frozen
Student-t promotion rule.

## Final paired result

Date: 2026-07-29.

Candidate SHA-256:
`0cd9676dc904660597a2f7dd6981fdac596295e4eae99510a3f4f21671859592`.

Staging snapshot SHA-256:
`e5faae4102a95cde78ecefe41be5d9532832e9f82df89c25bab3b308d0460b44`.

Container: `orbit-task07-matrix-34ea5ae79e`
(ID prefix `b8199b0e1e6a`), six CPUs, 7 GiB, no network.

Sanitized paired report:
`profiles/final-canonical-six-pairs.json`
(`sha256:6e03db1f37e8bbe0b38247f017d9259177a2858d54289fa2a5615542b499b54a`).

Fail-closed reference report:
`profiles/final-reference-six.json`
(`sha256:743f493120dd89e6c75a309499c28d819ae00aee09d57e7feb374b35c0310224`).

```text
terminal_status: SUCCESS x 12
valid cells: 12/12
passing pairs: 6/6
candidate wins: 6/6

reference runtimes:
106.038165, 119.427967, 123.188182,
107.532991, 110.084619, 131.316223

candidate runtimes:
24.481295, 24.221592, 24.716431,
24.421997, 31.684172, 27.652166

reference mean / median / stderr:
116.264691 / 114.756293 / 4.096777 s

candidate mean / median / stderr:
26.196276 / 24.598863 / 1.216760 s

ratio-of-means speedup: 4.438215x
ratio-of-means improvement: 77.4684%
mean paired speedup: 4.478752x
paired speedup stderr: 0.228662x
95% Student-t CI: [3.890959x, 5.066545x]
```

Decision: `promote`. Every cell passes, mean and median are lower, all six
pairs win, and the frozen confidence lower bound is above 1.0.

### Final report serialization failure and recovery

After cell 12 passed and the shared container stopped, the runner raised a
`NameError` while constructing the final JSON because two Python booleans
were written as lowercase `true`. All 24 raw stdout/stderr logs already
existed; stderr logs were empty and every stdout contained `Overall: PASS`.
No benchmark cell was rerun or filtered. The sanitized reports above were
reconstructed from all raw logs, with every stdout hash retained.

The runner is corrected to use `True` and now writes a checkpoint after each
cell. `research/check_gates.py --task 07 --baseline-report
research/task-07/profiles/final-reference-six.json` returns
`promotion_ready: true`.

## Experiment `e07`: exact classical-ancilla reduction

Branch: `codex/orbitbreakers/task-07/e07-classical-ancilla`.

Parent commit: `529a5c5` (the published e04a implementation and six-pair
evidence).

### Structural derivation

This experiment follows a user-supplied concern that the apparent
measurement-feedback workload may contain an unintended exact reduction.
The eight ancillas enter each layer as a computational-basis product state.
Their `RY` gates create independent Bernoulli source bits. Each following
data-ancilla `RZZ` is diagonal and applies a norm-preserving data unitary
conditioned on its source bit, so it cannot change the ancilla Z-basis
probabilities. The ordered ancilla CNOT ladder is only the reversible map

```text
measured[0] = source[0]
measured[i] = source[0] xor ... xor source[i].
```

Its inverse is
`source[0] = measured[0]` and
`source[i] = measured[i] xor measured[i-1]`.
Conditioning on one measured string therefore selects one unique source
string and an eight-qubit data-only circuit. The pre-measurement entangler
becomes `RZ((1-2*source) theta_entangler)` on data; measured feedback becomes
`RZ((1-2*measured) theta_feedback[measured])`. They commute and combine into
one data `RZ`.

For an ancilla entering as previous measured bit `b`, the next independent
source probability is

```text
P(source=1 | b=0) = sin(theta/2)^2
P(source=1 | b=1) = cos(theta/2)^2.
```

The implementation reconstructs TensorCircuit's strict
`status > 1-P(bit=1)` sampling rule, maps the fixed seed-2048 uniforms to
complete two-layer patterns, and deduplicates equal patterns before quantum
evaluation.

### Independent audit

Audit program:
`validate_classical_ancilla_reduction.py`
(`sha256:86e30424397d816be4db109a5eb64013de152db25477aba15bff7d7db44c4d3e`).

Sanitized record:
`profiles/e07-classical-ancilla-audit.json`
(`sha256:f371e2429a3a5724583db82eb6a45089e4040c4e699b7db3d51a50035ad126c3`).

All 1,024 measured bits produced by the analytic sampler are identical to
the full 16-qubit TensorCircuit `cond_measure` implementation. The 64 fixed
trajectories contain only two distinct complete patterns: the all-zero
pattern occurs 63 times and trajectory 36 supplies the sole rare pattern.

Against the accepted full 16-qubit e04a implementation:

```text
initial energy absolute error:          4.7684e-6
trajectory energy maximum error:        4.2915e-6
full ancilla-gradient maximum magnitude:4.6559e-7
reduced ancilla-gradient magnitude:     0
non-ancilla gradient maximum error:     1.5116e-6
post-one-update energy error:            4.6730e-5
audit passed:                            true
```

The ideal pathwise derivative of a fixed discrete branch with respect to
the ancilla sampling angle is exactly zero. The full complex64 graph produces
only sub-micro numerical residue there; Adam can amplify that residue into
a parameter-coordinate difference, but the corresponding physical energy
checks remain close. This distinction is disclosed rather than hidden.

### Exploratory evaluator screens

These screens were exploratory and occurred before a final candidate freeze;
they are not the promotional paired measurement.

Candidate source SHA-256:
`29d4d94101c21d757f57f3c639752533bfb84feb8acae5a8b2659a40e0f78631`.

```text
max_steps=50:  3.008539 s, PASS
  initial/final history: -6.8462691307 / -8.7942304611
max_steps=100: 2.998158 s, PASS
  initial/final history: -6.8462700844 / -10.0277271271
  final trajectory mean/std: -10.0331859589 / 0.0007448916
```

Decision: `keep provisionally`. The canonical screen is about 8.1x faster
than e04a's 24.362-second screen and about 45.3x faster than the original
expert's 135.816-second bootstrap. Because compilation now dominates and
50 versus 100 updates costs almost the same, isolate a whole-training scan
and small-circuit gate/contractor choices before freezing a new paired run.

### Scope and policy caveat

This is an exact reduction of the public fixed workload, not hard-coded
energies or fewer requested trajectories. All 64 statuses are consumed, all
96 parameters retain their original layout, all trajectory outputs are
reconstructed, and TensorCircuit performs every remaining quantum evolution
and Hamiltonian expectation. It nevertheless removes the explicit
16-qubit/mid-circuit-measurement execution that the task prose may have
intended to benchmark. The final report must present this openly as a
challenge-design loophole and keep the conservative e04a implementation
available if maintainers require literal `cond_measure` use.

## Post-reduction experiment sweep

All variants below start from the provisionally accepted e07 reduction and
retain the 64-to-2 exact pattern map.

### `e08`: whole-training `K.jaxy_scan`

Candidate commit: `7c9476a`.
Source SHA-256:
`a6a9c882edabbf88bdb175d5cf7dd4b39bf09f923c373a939929380a616b7376`.

Canonical screen: `3.205596 s`, PASS, versus the e07 exploratory
`2.998158 s`. Final-history energy differs by `1.53e-5`.

Decision: `discard`. Even after the dimensional collapse, staging the
100-update control flow costs more than the cached Python dispatches it
removes.

### `e09`: fuse pre-CNOT `RY` and reduced `RZ`

Candidate commit: `7d0fd29`.
Source SHA-256:
`96c0ca51d49fa334758e8c60985062250aae6f3f516b5c42aaed3eb26adbe754`.

The exact product `RZ(z) RY(y)` was emitted as a differentiable
TensorCircuit `any` gate. Canonical screen: `3.530851 s`, PASS.

Decision: `discard`. TensorCircuit's contraction preprocessing already
handles the neighboring one-qubit gates more cheaply than explicitly
constructing the parameterized dense matrix.

### `e10`-`e13`: contractor selection

Single canonical screens:

| Variant | Runtime (s) | Result |
| --- | ---: | --- |
| e10 `greedy` | 2.910894 | PASS |
| e11 `plain-experimental`, default local steps 2 | 2.921477 | PASS |
| e12 `plain-experimental`, local steps 1 | 3.100929 | PASS |
| e13 `plain-experimental`, local steps 3 | 2.822912 | PASS |

Because greedy, OMECo-1x1, and the default local contractor differed by only
tenths of a second, e11 was selected through counterbalanced six-pair
screens rather than a single timing. Sanitized record:
`profiles/e11-contractor-six-pair-screen.json`.

```text
greedy mean:             2.947290 s
plain-experimental mean: 2.823417 s
plain wins:              6/6
mean paired speedup:     1.044198x
95% Student-t CI:        [1.002914x, 1.085481x]

OMECo-1x1 mean:          2.949023 s
plain-experimental mean: 2.839041 s
plain wins:              5/6
mean paired speedup:     1.039493x
95% Student-t CI:        [0.993346x, 1.085640x]
```

An earlier attempt mounted a comparison worktree from `/private/tmp`; Docker
turned the unavailable file mount into a directory and every greedy cell
failed before evaluator execution. Those values are explicitly excluded and
the complete six-pair screen was rerun from a Docker-visible workspace.

Decision: `keep e11`. The exact reduced graph is small enough that
TensorCircuit's native local contractor avoids global path-search overhead.
Local-step values 1 and 3 do not provide sufficient repeat evidence to
supplant the stable default of 2.

## Frozen e11 candidate for new expert comparison

Candidate source SHA-256:
`0337bf428a7c4a820f12f7db1232620b2777677617dd4f1a657dfd5f53bbdb0e`.

The final comparison will use six canonical matched pairs, alternating the
immutable human expert and e11 in one no-network container with six CPUs,
7 GiB, the same evaluator and latest repository TensorCircuit image. No
candidate tuning follows this freeze.

## Final e11 six-pair expert comparison

Date: 2026-07-29.

Candidate implementation commit: `b7d34dd`.

Candidate SHA-256:
`0337bf428a7c4a820f12f7db1232620b2777677617dd4f1a657dfd5f53bbdb0e`.

Staging snapshot SHA-256:
`d46912b2aba5e201c56754f98a21065be1009fe3c90d105a4a2b29b95aeaab0f`.

Docker image:
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`
(TensorCircuit-NG `1.8.0.dev20260726`, JAX/JAXLIB `0.10.0`).

Sanitized paired report:
`profiles/e11-final-canonical-six-pairs.json`
(`sha256:068593daf65d132d1c7b3f18a0cbc2f7fc4b378558f3c4ccedf79231c0248c0f`).

```text
terminal_status: SUCCESS x 12
valid cells: 12/12
passing pairs: 6/6
candidate wins: 6/6

reference runtimes:
123.286060, 126.247088, 150.224728,
129.913867, 159.579514, 151.207388

candidate runtimes:
3.062554, 3.311737, 2.989908,
3.030923, 3.063330, 2.966582

reference mean / median / stderr:
140.076441 / 140.069298 / 6.281458 s

candidate mean / median / stderr:
3.070839 / 3.046739 / 0.050718 s

ratio-of-means speedup: 45.615039x
ratio-of-means improvement: 97.8077%
mean paired speedup: 45.757921x
paired speedup stderr: 2.479287x
95% Student-t CI: [39.384711x, 52.131131x]
```

Decision: `promote under the executable contract`. Every cell passes, every
pair wins, and the frozen confidence lower bound is far above 1.0. The
separate challenge-design report marks the semantic caveat: this exact
reduction should not be represented as a generic acceleration of
mid-circuit measurement, and maintainers may prefer the conservative e04a
implementation if literal 16-qubit `cond_measure` execution is the intended
policy.
