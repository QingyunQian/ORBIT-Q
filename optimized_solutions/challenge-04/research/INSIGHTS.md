# Task 04 Research Insights

Task: `task-04`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment `e06-fused-rxx-kraus`, source SHA-256
`251f283d208ebb5316c8347ff71ea6da66aca5fd11b42f25032db32ada83ef0c`.
Against the accepted paired-Kraus parent, six paired Docker runs give
`1.104347x ± 0.021593x`, 95% CI `[1.048841x, 1.159853x]`, 6/6 wins and
12/12 cells PASS. The retained cumulative factors are probe VMAP, exact
pairing of adjacent Kraus nodes, and absorption of RXX into those nodes.

The fresh final cumulative comparison against the immutable expert gives
expert `14.742286 ± 0.274987 s`, campaign best `5.672258 ± 0.148840 s`,
paired speedup `2.602133x ± 0.028270x`, 95% CI
`[2.529463x, 2.674804x]`, and 6/6 wins. See
[`IMPLEMENTATION_COMPARISON.md`](IMPLEMENTATION_COMPARISON.md).

## Preserved semantics

- Four fixed 12-qubit probe preparations and the exact even/odd `RXX(0.31)`
  brickwork order.
- User-defined asymmetric bit-flip channel represented by the specified three
  differentiable Kraus tensors and applied independently after each
  entangler.
- Target observables generated inside `run_solution` from true probabilities.
- All 12 single-Z values plus full-chain Z parity for every probe.
- Sigmoid probability parameterization, fixed initialization, exactly 120
  Adam updates at learning rate `0.04`, and pre-update loss history.
- Complex64 TensorCircuit-NG central computation and unchanged NumPy output
  schema.

## Confirmed bottlenecks

- In the measured TensorCircuit-NG build, `tc.DMCircuit` and
  `tc.DMCircuit2` both resolve to
  `tensorcircuit.densitymatrix.DMCircuit2`; each Kraus list becomes a local
  superoperator TN node. Renaming the class cannot change runtime.
- A 12-qubit complex64 density matrix occupies 128 MiB before gradient
  temporaries; one observable table invokes 88 one-qubit channels.
- The exact canonical reference takes `14.671115 ± 0.123236 s` end to end.
- The tracked profile measures `2.629 s` target construction, `3.068 s`
  lowering, `10.712 s` XLA compilation, but only `1.257 ms` per compiled
  update (`0.151 s` projected for all 120). Compilation of the 21,720-line
  StableHLO graph, not Python optimizer dispatch, dominates this workload.

## What worked

Building the four exact probe states once with `tc.Circuit` and batching the
identical noisy density-matrix circuit through TensorCircuit `K.vmap` reduced
StableHLO from 21,720 to 4,855 lines, lowering from `3.068` to `0.616 s`, and
XLA compilation from `10.712` to `2.014 s`. This dominates the increase in
steady per-update execution from `1.257` to `8.035 ms`.

Replacing the two single-qubit channel nodes after each entangler with the
exact nine-Kraus product channel halves channel nodes from 22 to 11 per probe.
Relative to probe VMAP alone, target construction falls from `2.323` to
`1.790 s`, lowering from `0.616` to `0.547 s`, and StableHLO from 4,855 to
4,602 lines. The paired benchmark attributes a further `1.053233x` speedup.

Absorbing each fixed RXX matrix into the nine product Kraus matrices removes
all 11 explicit RXX nodes per probe. Relative to paired Kraus alone, StableHLO
falls from 4,602 to 3,283 lines, lowering from `0.547` to `0.396 s`, and
compilation from `2.138` to `1.639 s`. Its independent paired speedup is
`1.104347x`.

## What did not work

The first baseline command session was interrupted after two passing cells and
did not produce a complete report. Those timings are excluded rather than
spliced into the complete baseline.

Changing only `tc.DMCircuit` to `tc.DMCircuit2` is a byte-level source edit
but a runtime no-op because both names are the same class in the measured
image. Its inconclusive `1.0180x ± 0.0159x` result is timing noise, not an
algorithmic factor.

Putting all 120 Adam updates in one TensorCircuit `K.jaxy_scan` is also exact
but regressed relative to the accepted VMAP parent: e02 `6.645754 ±
0.033795 s` versus e03 `6.736220 ± 0.061509 s`, paired speedup `0.986981x ±
0.010315x`, with only 2/6 wins. This confirms the profile's prediction that
host dispatch is not the useful bottleneck after probe batching.

TensorCircuit's default expectation reuse is unsafe for the canonical
differentiated batch at the frozen 7 GiB limit. It is exact on the reduced
case, but the 12-qubit smoke attempts a 9,263,654,532-byte allocation and
fails before timing. Explicit `reuse=False` must remain in the accepted
implementation.

Replacing nested TensorCircuit stack/cast construction with static complex64
matrix units is exact but slower and more variable: `0.969277x ± 0.035507x`
relative to e06, with only 3/6 wins. Source line reduction is not evidence of
a better XLA graph.

## Open hypotheses

1. Exact MPO/purification formulations through TensorCircuit-native classes if
   simpler delayed contraction is insufficient.
2. Measurement batching only if it preserves the low-memory scalar
   contractions that `reuse=False` provides.

## Evidence limits

- Claims are limited to the canonical fixed Task 04 workload and current host.
- No external matched runtime establishes global state of the art.
- Runtime binding and local-node behavior are specific to the measured
  TensorCircuit-NG development image; older public class bodies differ.
