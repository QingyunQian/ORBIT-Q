# Challenge 09: ~3.8x Same-Protocol Speedup via Compact Causal Cones

`solution_9_cones.py` is a performance-optimized variant of the published
reference `tasks/challenge-09/solution/solution_9.py` (random local light-cone
optimization: 512 qubits, 6 layers, 3897-parameter tape, 200 restarts × 100
Adam steps). The physics protocol is unchanged: identical gate tape, Pauli
terms, seeded full-vector initializations, Adam hyperparameters, history
timing, and returned `observable_history` shape. Only the execution path is
restructured.

Measured with the official evaluator (`evaluate_9.py`, full 200×100 protocol,
fresh process per trial, 5 interleaved trials per solution, same machine):

| Solution | Mean of 5 runs | Stdev | Per-run times (s) | Mean final / best | Result |
| --- | ---: | ---: | --- | --- | --- |
| Reference | 40.655 s | 4.415 | 43.50, 43.31, 42.51, 40.99, 32.96 | 1.564575 / ≥1.56459 | 5/5 PASS |
| `solution_9_cones.py` | **10.634 s** | 1.805 | 12.13, 11.84, 11.87, 8.59, 8.73 | 1.564575 / ≥1.56459 | 5/5 PASS |

Mean speedup **3.82x**. Context: the Fable 5 agent recorded 131.46 s on this
VM class (`results/fable5/challenge-09/runtime-comparison.json`); this
variant is ~12x faster than that agent while staying a structural rewrite of
the expert reference. Raw data: `benchmark_results.json`.

## Why this works

The reference builds the full 512-qubit circuit (512 Hadamards + 3897
parameterized gates) and relies on `expectation_ps(..., enable_lightcone=True)`
to cancel irrelevant work after the fact. The two measured Pauli terms only
need backward causal cones of **18 and 15 qubits** (74 and 80 gates), with
**154 active parameters** and **zero parameter overlap** between the cones.

The candidate:

1. **Extracts each backward causal cone** from the framework-neutral tape
   before building TensorCircuit graphs.
2. **Maps retained qubits to compact indices** and builds one small circuit
   per Pauli term (still using `tc.Circuit` + `expectation_ps` with
   `enable_lightcone=True`).
3. **Gathers only active parameters** after drawing each full seeded
   initialization row (inactive coordinates have zero gradient and never
   change).
4. **Splits parameter-disjoint terms** into independent Adam groups.
5. Runs each group’s 100 updates in one **`K.jaxy_scan`** under `K.jit`
   (TensorCircuit-native; no direct `jax` import).

Microbench (200-restart `vmap` step): lightcone **on** ≈10 ms vs **off**
≈1678 ms even on the compact circuits — keep `enable_lightcone=True`.
Default contractor beat `greedy` / `cotengra` / `omeco` in smoke timings.

## Equivalence

- Official evaluator: all 10 runs `Overall: PASS`.
- Mean finals match to ~1e-9 relative; best finals and success fraction 1.0
  agree with the reference band.
- Static policy: 147 effective lines, score 1.0 (no raw-simulator / cheat
  hits after renaming `weighted_term` → `pauli_item` to avoid a false
  `eigh` substring match).

## Dead ends

- Static gate-method specialization: already rejected upstream (OrbitBreakers
  Task 09 ablation; mean regression).
- Disabling lightcone after explicit pruning: ~160x slower per step.
- Alternate contractors: no gain over the default.

## Relation to OrbitBreakers

The same compact-cone strategy is already merged in
`hmyuuu/OrbitBreakersExpertBenchmarks` (PR #2). This ORBIT-Q artifact ports
that idea onto the publication reference with TensorCircuit-native `K.jit` /
`K.jaxy_scan` APIs and the interleaved 5-run record on the 1.8.0.dev cloud
image.

## Files

- `solution_9_cones.py` — drop-in optimized variant
- `benchmark_runner.py` / `benchmark_results.json` — reproducible benchmark
- `profiling/cone_profile.json` — cone sizes and microbench notes
