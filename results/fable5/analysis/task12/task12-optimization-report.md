# Task 12 expert-solution optimization: 2.38× speedup (10.30 s → 4.32 s)

Target: the human-expert reference for ORBIT-Q task 12 (variational circuit to
MPS overlap, 32 qubits, 2 SU(4) brickwork layers, exactly 5000 Adam updates).
Environment: the canonical benchmark image
`challenge-benchmark-quantum-tensorcircuit:py311` (tensorcircuit-nightly
1.8.0.dev20260726, jax 0.10.0, omeco 0.2.4), official `evaluate_12.py`,
evaluator-reported `run_solution` time, fresh container per run.

## Headline (5 counterbalanced pairs, all runs `Overall: PASS`)

| Implementation | Runtime (5-run mean ± sd) | Speedup |
| --- | ---: | ---: |
| Expert reference | 10.301 ± 0.242 s | 1.00× |
| **Optimized** | **4.320 ± 0.030 s** | **2.38× (−58.1%)** |

Physics is unchanged: same ansatz, same SU(4) parametrization (identical
generator basis and parameter order as `tc.gates.su4_gate`), same seed and
Adam schedule, same direct MPS-bra × circuit-ket overlap loss. Final fidelity
0.8693 vs the reference's 0.8699 (both ≥ 0.85 threshold; tiny difference from
floating-point reordering only).

## Where the time went (profiling)

Two-point profiling (500- vs 3000-step runs) of the reference gave
per-step ≈ 0.74 ms and fixed cost ≈ 5.8 s, i.e. at 5000 steps the wall time is
roughly **60% one-time cost (JIT compile + contraction-path search) and 40%
stepping** — a very different profile from the physics-dominated tasks. The
levers are therefore graph size, path-search cost, and dispatch overhead, not
arithmetic.

A caution for others profiling this task: in-process multi-run timing is
misleading because tensorcircuit caches contraction paths between traces; the
first extrapolation suggested a 1.31× win that shrank to 1.05× under fresh
processes. All numbers above are fresh-process, official-evaluator runs.

## The three changes (ablation, single-shot official runs)

| Variant | Runtime | Δ vs previous |
| --- | ---: | --- |
| Reference (per-gate `su4`, Python loop, omeco) | ~10.3 s | — |
| + `jax.lax.scan` over the 5000 updates | 9.55 s | −0.7 s: removes 5000 Python dispatches |
| + cheap deterministic-budget cotengra greedy paths | 8.77 s | −0.8 s: omeco's stochastic search costs seconds per process; a bounded greedy search (max_repeats=32, max_time=1 s) finds an equally good path for this quasi-1D network |
| + **batched SU(4) construction** | **4.32 s** | **−4.5 s: the dominant win** |

The batched-SU(4) change: the reference builds each of the 31 gates with its
own 15-term generator sum and `expm`, producing 31 small expm subgraphs that
XLA compiles and differentiates separately. Building all generators with one
`einsum('gk,kab->gab', theta, paulis)` and one batched
`jax.scipy.linalg.expm` collapses these into a single vectorized op — the
compile graph shrinks sharply and the backward pass vectorizes, cutting both
the one-time and per-step cost. Gate semantics are bit-for-bit the same
parametrization (same Pauli basis order as `tc.gates.su4_gate`).

## Files

- `solution_12_optimized.py` — drop-in replacement measured above (protocol
  contract unchanged: same config keys, same returned dictionary).
- Raw pair timings: expert 9.904/10.474/10.453/10.438/10.235 s; optimized
  4.342/4.276/4.344/4.300/4.336 s.

## Suggested upstream framing

For OrbitBreakersExpertBenchmarks: drop `solution_12_optimized.py` into
`src/solutions/task-12/solution_12.py` and run
`./bench run 12 --solution optimized --compare-to reference --repeat 6`.
The three changes are independent; if a minimal diff is preferred, the batched
SU(4) construction alone captures most of the gain.
