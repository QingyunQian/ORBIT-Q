# Task 05 research insights

Task: `task-05`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

The accepted implementation is the exact no-QR MPS candidate:

- source: [`src/solutions/task-05/solution_5.py`](../../src/solutions/task-05/solution_5.py);
- SHA-256:
  `e1a0d8a13020687f0afc89867e114683c052200c955a3515c80397a6a580b24e`;
- candidate mean: `6.898409 s`;
- immutable expert mean in the matched session: `97.083712 s`;
- mean paired speedup: `14.075570x`;
- 95% Student-t interval: `13.675752x–14.475387x`;
- candidate wins: `6/6`;
- passing cells: `12/12`.

This is a valid same-machine improvement on the public fixed Task 05
workload, not a cross-machine absolute-runtime or global SOTA claim.

## Decisive structure

- Every `exp(a X)` is one-site and leaves MPS ranks unchanged.
- `exp(b Z.Z) = cosh(b) I.I + sinh(b) Z.Z` has exact operator-Schmidt rank 2.
- Each brickwork bond is used five times, so the exact state bond dimension is
  at most 32.
- Generic QR/RQ canonicalization is unnecessary when this exact rank bound is
  known. Applying TensorCircuit's local MPO-times-MPS kernel without QR/RQ
  converts the historical MPS timeout into a `6.9 s` complete run.
- Ten differentiable norm contractions, the bond-3 TFIM expectation, and all
  600 Adam updates remain in the computation.

## Correctness findings

- Dense versus reconstructed state maximum error: `2.79397e-08`.
- Norm error: `4.17233e-07`.
- Initial energy-density error: `1.79052e-04`.
- Gradient maximum error: `1.19507e-05`.
- Non-degenerate one-update parameter error: `3.72529e-09`.
- Maximum observed exact bond dimension: 32.

The first uniform RX parameter has an analytically zero gradient on
`|+>^18` after normalization. Do not use raw first-step Adam parameter
difference at that one degenerate coordinate as an equivalence criterion;
retain state, energy, full gradient, conditioned update, and full evaluator
checks.

## Measured factor decisions

- **Keep: exact bounded-rank no-QR MPS.** The complete candidate achieves
  `14.075570x` versus the immutable expert.
- **Discard: dense two-site layer fusion.** Mean `1.022847x`, 95% interval
  `0.983443x–1.062251x`; fewer nodes did not establish a gain.
- **Discard: RX absorption into both MPS MPO branches.** Mean `0.843725x`,
  95% interval `0.822186x–0.865264x`; it duplicated local matrix work and
  regressed all six pairs.
- **Historical keep: whole-training scan.** The earlier pinned campaign
  isolated `1.098033x`.
- **Historical keep: reusable greedy contractor on the dense graph.** It
  helped the old dense candidate, but the final MPS no longer imports or uses
  Cotengra.

## Attribution boundary

The final direct exact-MPS-versus-accepted-dense-parent run did not start
because the Docker approval service disconnected and prohibited automatic
retry. Do not estimate a paired factor result from separate sessions. The
eligible current claim is the full MPS candidate versus the immutable expert;
historical scan/contractor ablations remain contextual.

## Do not repeat unchanged

- Generic `MPSCircuit.apply_MPO` with QR/RQ.
- `K.jaxy_scan` as a wrapper-only substitution on the old dense graph.
- `plain-experimental` contractor.
- Algebraic contraction primitives on the dense circuit.
- Single-array dense parameter layout.
- Dense layer gate fusion without a new contraction-path mechanism.
- RX absorption into both MPS MPO branches.

## Reusable lesson

Before tuning contraction search on a dense state, derive exact rank growth
from the circuit's operator-Schmidt structure. A bounded-rank representation
can dominate path and dispatch micro-optimizations by an order of magnitude.
When a framework's generic MPS method canonicalizes after every update,
inspect whether the rank is already statically bounded; if so, its local
TensorCircuit contraction kernel may be used exactly without QR/RQ.
