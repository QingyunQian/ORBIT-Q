# Challenge 11: 1.43x Same-Protocol Speedup of the Reference Solution

`solution_11_fused.py` is a performance-optimized variant of the published
reference `tasks/challenge-11/solution/solution_11.py` (spin-1 Haldane-chain
VQE: 12 three-level sites, dense 3^12 state, 5 brickwork layers, exactly 500
Adam updates, string-order readout). The physics protocol is unchanged:
identical parameter layout and seeded initialization, identical layer
structure (per-site rz/ry/rz rotations, then even- and odd-bond spin-1
entanglers `expm(-i[theta S.S + (phi-theta) SzSz + beta (S.S)^2])`),
identical Adam schedule on the identical energy density, identical returned
quantities. Only the computation is restructured.

Measured with the official evaluator (`evaluate_11.py`, full 500 steps,
fresh process per trial, 5 interleaved trials per solution, same machine):

| Solution | Mean of 5 runs | Stdev | Per-run times (s) | Energy gap / string MAE | Result |
| --- | ---: | ---: | --- | --- | --- |
| Reference | 168.574 s | 1.298 | 170.25, 167.31, 167.35, 168.48, 169.48 | 0.0691 / 0.0552 | 5/5 PASS |
| `solution_11_fused.py` | **118.200 s** | 3.735 | 115.73, 116.68, 114.45, 120.77, 123.37 | 0.0692 / 0.0551 | 5/5 PASS |

Mean speedup **1.43x**. Both solutions land on the same optimization quality
to three decimals (energy-density gap 0.0691 vs 0.0692, string-order MAE
0.0552 vs 0.0551, thresholds 0.12/0.12). For context, the Fable 5 agent
solution recorded 134.33 s on this VM class
(`results/fable5/challenge-11/runtime-comparison.json`); this variant is
~1.14x faster than that while staying a minimal-diff edit of the expert
reference. Raw data: `benchmark_results.json`.

## Why this workload is different from challenge 12

Stage-split profiling (`profiling/profile_reference.py`) shows the reference
spends essentially everything in the step loop, not in compilation:

| Stage | Reference | Share |
| --- | ---: | ---: |
| jit trace + XLA compile (HLO ~8100 lines) | 4.0 s | 2% |
| 500-step loop (319.5 ms/step) | 159.7 s | 97% |
| post-training block (eager) | ~0.3 s | <1% |

Inside the step: forward `build_state` 80.2 ms (5 layers x 47 gate
applications on the dense 4.25 MB state), forward energy 35.9 ms (23
separate `expectation` contractions), backward ~200 ms. Per-gate cost is
memory-bandwidth-bound and strongly position-dependent
(`profiling/proto_micro.py`: a 9x9 two-qudit contraction costs 0.45 ms at
the last bond but up to 2.9 ms at middle bonds - the strided-axis
transpose+gemm lowering, not FLOPs, dominates).

## The changes (all exactly algebra-preserving)

1. **Exact gate fusion, 47 -> 11 state passes per layer.** The three
   single-site rotations per site are composed into one 3x3 unitary
   (batched matrix products over all 12 sites), and because the even bonds
   tile all sites, each pair's composite is absorbed into its even-bond
   9x9 entangler: `even'_k = even_k @ (u_{2k} (x) u_{2k+1})`. Each layer
   then applies 6 fused even + 5 odd two-qudit unitaries - the identical
   layer unitary up to float rounding.
2. **Batched fixed-order entangler exponentials.** All 9x9 generators of a
   layer are exponentiated in one batched fixed 2^5 scaling-and-squaring
   diagonal Pade(3,3) pass (exactly unitary for anti-Hermitian input).
   The largest generator spectral-norm bound over all 500 steps is 2.96,
   i.e. 0.093 after scaling, far below the approximant's accuracy edge
   (`profiling/diag_equivalence.py`).
3. **Diagonal single-ion term via one weight vector.** The 12 separate
   one-site `expectation` calls for `D * sum_i (S_i^z)^2` (a diagonal
   observable) are replaced by one precomputed basis-weight vector and a
   single `sum(w * |psi|^2)` pass; the 11 non-diagonal bond terms keep the
   framework expectation call per bond. Forward energy drops 35.9 -> 15.4 ms.
4. **`jax.lax.scan` over the 500 Adam updates and a jitted post-training
   block** (the reference recomputes the final state, energy, and string
   orders eagerly).

State evolution and all non-diagonal observables still go through
`tc.QuditCircuit` exactly as in the reference.

## Equivalence evidence

- From the identical seeded initialization, the per-step energy deviation is
  2.8e-5 after 5 steps and 8.0e-5 after 100 steps (complex64 noise on
  energies of order 0.5; fusion changes rounding order, so deltas start at
  the noise floor rather than zero and grow only through optimizer
  dynamics).
- Final metrics agree to the third decimal on every trial (see table), well
  inside the evaluator's 0.12 thresholds and each other's run-to-run
  scatter; all 10 benchmark runs print `Overall: PASS`.
- `static_policy.py` scores the file 1.0 (191 effective lines, no
  raw-simulator or cheating hits).

## What did not work (measured dead ends)

| Attempt | Result | Why |
| --- | --- | --- |
| Unrolling the 5 layers instead of `K.scan` | 295.5 ms/step vs 264.6 | larger compile, worse scheduling |
| Splitting even/odd into separate circuits | 285.3 ms/step | one materialization more per layer |
| Fusing 4 sites into dim-81 blocks | rejected on FLOPs | 81x81 crossing gates turn a bandwidth problem into a compute problem (cf. challenge-12 dim-16 result) |
| Bare reshape-matmul layer application (floor probe) | 8.25 vs 14.33 ms/layer forward | only ~1.7x headroom even abandoning the framework circuit path; not shippable under framework-fidelity rules and not pursued |

The dead-end table quantifies the ceiling: this workload is
memory-bandwidth-bound on a dense 3^12 state, so the achievable gain comes
from cutting state passes (47 -> 11 per layer, 24 -> 2 for the diagonal
observable), not from compilation or dispatch, which is why the speedup is
1.43x here versus 4.2x on the compile-bound challenge 12.

## Environment and reproduction

Same setup as `optimized_sloutions/challenge-12/REPORT.md`: cloud VM, 4 vCPU
Intel Xeon x86_64, 15 GB RAM, Python 3.12.3 venv with the pinned
`frameworks/tensorcircuit/requirements.txt` set (tensorcircuit-nightly
1.8.0.dev20260726, jax/jaxlib 0.10.0, optax 0.2.8). Both arms ran interleaved
in the identical environment (same-machine/same-environment T/T_ref
convention).

```bash
cd optimized_sloutions/challenge-11
/path/to/.venv-c12/bin/python benchmark_runner.py 5
```

## Upstream porting notes

The file is a drop-in replacement for the challenge-suite reference
`solution_11.py`: same `run_solution(config)` contract, same returned keys,
no new dependencies. The fusion generalizes to any even site count with the
same brickwork layout; the diagonal-onsite weight vector is rebuilt from
`config` at run time.
