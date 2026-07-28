# Task 11 speed analysis: why the Fable 5 solution is faster, and how to go further

Scope: challenge-11 (spin-1 Haldane-chain VQE). This report (1) explains, with
profiling evidence, why the Fable 5 candidate outruns the human-expert
reference, and (2) profiles the bottleneck and demonstrates a further ~2x
speedup that also applies to improving the expert implementation.

All numbers are from one local environment (Python 3.12, tensorcircuit-nightly
1.8.0.dev20260726, jax 0.11.0, `JAX_ENABLE_X64=0` so both sides are complex64,
`NUMBA_DISABLE_JIT=1`), each figure from a two-point fit (20 and 120 steps) that
separates per-step cost from fixed compile+finalize cost.

## 1. Why the candidate is faster

Both implementations share the same physics skeleton: `QuditCircuit(dim=3)`,
closed-form spin-1 `Rz`/`Ry`, an `expm` two-site entangler including the fixed
`beta (S.S)^2` term, Adam for 500 steps. The two structural differences are:

| | Expert reference | Fable 5 candidate |
| --- | --- | --- |
| Layer staging | `K.scan` over the 5 layers | Python-unrolled 5 layers |
| Energy | sum of 23 separate `circuit.expectation` contractions (11 bonds + 12 on-site) | same (23 contractions) |

Profiled cost:

| Implementation | per-step | fixed (compile+final) | projected 500-step |
| --- | ---: | ---: | ---: |
| Expert reference (scan layers) | 341.9 ms | 5.6 s | 176.5 s |
| Fable 5 candidate (unrolled layers) | 189.7 ms | 24.6 s | 119.4 s |

**The candidate is faster because Python-unrolling the layers exposes the whole
five-layer circuit to XLA as a single graph, which XLA fuses and optimizes
globally, cutting the per-step cost 1.8x (342 -> 190 ms).** The expert's
`K.scan` keeps the compiled graph small — so it compiles ~19 s faster (5.6 s vs
24.6 s fixed) — but scan blocks cross-layer fusion and reintroduces per-iteration
loop overhead, so each of the 500 steps is slower.

This is a classic compile-time-vs-run-time trade-off. `K.scan` is the right
choice when the step count is small or the layer count is large enough to make
the unrolled graph impractical to compile. Here, at 500 steps and only 5 layers,
the per-step win dominates: unrolling pays +19 s of compile to save
(342-190) ms x 500 = 76 s of stepping, a net ~57 s gain — exactly the observed
176.5 - 119.4 s gap.

## 2. Bottleneck and a further ~2x improvement

Profiling the per-step cost further: a forward energy evaluation is 130 ms and
its reverse-mode gradient is 369 ms (reference), i.e. the step is dominated by
differentiating the **energy evaluation**, which contracts the full
3^12 = 531441-dimensional state 23 times (once per Hamiltonian term).

Optimization: replace the 23 tensor-network contractions with **one sparse
matrix-vector product** against a precomputed Hamiltonian. The 12-site spin-1
`H` has only ~23 local terms, so as a sparse operator it has manageable nnz;
energy becomes `real(conj(psi) @ (H @ psi)) / n`, a single matvec whose gradient
is also a single (adjoint) matvec. The circuit still produces the state through
the framework (state preparation is unchanged and framework-native); only the
expectation is swapped from many contractions to one sparse matvec — the same
pattern the challenge-02/03 references use via `PauliStringSum2COO`.

Result (`solution_11_optimized.py`, functional PASS, energy gap 0.069, string
MAE 0.055 — identical physics to the candidate):

| Implementation | per-step | fixed | projected 500-step | speedup vs expert |
| --- | ---: | ---: | ---: | ---: |
| Expert reference | 341.9 ms | 5.6 s | 176.5 s | 1.00x |
| Fable 5 candidate | 189.7 ms | 24.6 s | 119.4 s | 1.48x |
| Optimized (unroll + sparse matvec) | 135.1 ms | 18.2 s | 85.7 s | **2.06x** |

The two optimizations stack: unrolling cuts per-step 342 -> 190 ms; the sparse
matvec cuts it further 190 -> 135 ms **and** lowers compile 24.6 -> 18.2 s
(one matvec subgraph instead of 23 contraction subgraphs).

## 3. How to improve the expert implementation

Both changes are drop-in improvements to the human-expert reference:

1. **Replace `K.scan` over layers with an unrolled loop.** For the fixed
   5-layer / 500-step workload this alone is ~1.5x. (Keep scan only if a much
   larger layer count makes the unrolled graph too expensive to compile.)
2. **Compute the energy as one precomputed sparse-`H` matvec** instead of 23
   per-term `circuit.expectation` contractions. This is the larger structural
   win and is orthogonal to the staging choice.

Applied together they reproduce the 2.06x measured here without changing the
physics, the ansatz, the optimizer, or the returned quantities.

## Reproduce

```bash
# from a venv matching the framework (tc 1.8 nightly, jax 0.11, diffrax):
python results/fable5/analysis/profile_twopoint.py <solution_file.py> <module>
python results/fable5/analysis/profile_forward_vs_grad.py reference <ref_file.py>
```

`solution_11_optimized.py` is the further-optimized variant measured above.
Absolute seconds are host-dependent; the per-step ratios and the structural
conclusions are not.
