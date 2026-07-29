# Task 10 Research Insights

Task: `task-10`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment `e02-noqr-mps`, commit
`34171c020c2f9613efefaddefb94dc87ec9cbf16`, is the campaign best.

The user-requested first five matched pairs on the same 6-CPU/7-GiB Docker
allocation measured:

- immutable expert mean: `18.931296 s`;
- candidate mean: `3.869287 s`;
- ratio-of-means speedup: `4.892709x`;
- ratio-of-means runtime reduction: `79.561424%`;
- mean paired speedup: `4.898251x`;
- 95% Student-t interval: `4.597784x–5.198719x`;
- paired wins: `5/5`, with every evaluator cell passing.

A sixth supplemental pair also passed and won. Across all six pairs, mean
paired speedup was `4.939894x` with 95% interval
`4.688793x–5.190995x`.

## Preserved semantics

- 22 qubits, the canonical 18 selected CMZ sites, two ansatz layers.
- Ordered `RX -> RZ -> RY` rotations followed by one exact CMZ per layer.
- Canonical initial basis state and NumPy seeded float32 parameters.
- Canonical open-boundary TFIM energy density.
- Exactly 200 Adam updates and pre-update energy history.
- Required NumPy keys, shapes, finiteness, and evaluator thresholds.
- TensorCircuit-NG remains the central quantum framework.

## Confirmed bottlenecks

- Source inspection shows the expert forms a general bra/MPO/ket tensor
  network and invokes OMECo during JAX tracing.
- The CMZ is intrinsically bond 2, and two exact applications permit an MPS
  bond-dimension bound of four; the generic contraction does not expose that
  simple sequential structure to the implementation.
- A fresh immutable-reference validation took 18.117244 seconds in the
  available newer ORBIT-Q image and passed.
- Phase-split profiling shows the 18.36-second reference is dominated by
  4.37 seconds of lowering/path search and 13.14 seconds of XLA compilation;
  the compiled optimizer step takes only 0.317 ms.
- The candidate lowers in 0.667 seconds and compiles in 2.873 seconds. Its
  complete compiled 200-step scan executes in about 39.68 ms. The
  end-to-end optimization is therefore still cold-compilation dominated.

## What worked

- Keep the state as exact TensorCircuit-NG MPS tensors. Two bond-2 CMZ MPOs
  give a known exact maximum bond dimension of four.
- Use the local TensorCircuit `MPSCircuit.apply_MPO` contraction kernel without
  QR/RQ. Since the bond rank is already statically bounded, canonicalization
  cannot compress the state.
- Evaluate the TFIM with a deterministic bond-3 MPO left-to-right contraction
  instead of tracing a general bra/MPO/ket network and OMECo path search.
- Compose TensorCircuit's `RX`, `RZ`, and `RY` gate matrices before applying
  the local MPS update.
- Put all 200 sequential Adam updates in one `K.jaxy_scan`.

Together these changes reduced first-five mean runtime by `79.56%` and passed
every functional check.

## What did not work

- The first pinned-image build failed at `apt-get update` after a network
  interruption. Retry rather than changing the environment lock.
- Prior ORBIT-Q evidence found a dense-state Task 10 implementation around
  18 times slower than the expert. Do not repeat that representation unchanged.
- Exact `MPSCircuit.apply_MPO` with QR/RQ canonicalization matches the first
  five optimization steps closely but diverges into a failing basin by step
  200. Explicit unfused rotations and complex128 intermediates both worsened
  the numerical trajectory.
- The exact-lock image detour was unnecessary for this campaign. The final
  relative comparison deliberately used the latest available TensorCircuit
  nightly for both implementations.
- An exact explicit product-branch implementation was numerically sound but
  slow to compile. Materializing all X/ZZ products took `16.063 s` for five
  updates versus e02's `3.804 s`.
- Replacing those products with one exact 4x4 observable transfer scan improved
  the branch prototype to `13.321 s`, still `3.50x` slower than e02. The
  product-branch observation is useful algebraically but not a win under this
  cold-compilation evaluator.

## Open hypotheses

1. Persistent compilation caching could help repeated applications, but the
   evaluator intentionally measures cold fresh processes, so it is outside the
   present comparison contract.

## Post-publication factor ablation

Five-pair removal ablations show that the whole-training scan is neutral on
this cold-compile-dominated workload: removing it changed the mean from
`3.809923 s` to `3.752754 s`, with only 2/5 promoted-candidate wins and a
paired-ratio 95% interval crossing one. It must not be credited with the
headline speedup.

Removing local `RX -> RZ -> RY` fusion increased the mean from `3.856303 s`
to `4.051055 s`; the fused implementation won 5/5 pairs, but the paired-ratio
interval still crossed one. Treat fusion as a small secondary improvement.

The performance claim should therefore attribute the dominant gain to the
exact bounded-rank MPS/MPO representation and its much smaller cold
lowering/compilation graph. The coupled CMZ-MPS and TFIM-MPO pieces were not
assigned fictional independent percentages.

## Evidence limits

- Same-host, same-container evidence for one canonical public workload only.
- Latest TensorCircuit nightly `1.8.0.dev20260726`, not the repository's older
  exact dependency lock.
- No memory profile, canonical scaling study, or matched external SOTA
  comparator.
- The result is the campaign best over the bundled human expert, not a global
  SOTA claim.
