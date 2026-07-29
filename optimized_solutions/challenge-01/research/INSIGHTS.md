# Task 01 Optimization Insights

This file records hypotheses and measured conclusions separately from the final
comparison report.

## Starting observations

- The immutable expert is already tensor-network based and never materializes
  a dense `2^32` state, but it asks a generic circuit contractor to process the
  same shallow one-dimensional structure on every objective evaluation.
- The loss creates 63 Pauli-pattern contractions. A direct TFIM MPO has bond
  dimension three and is an exact algebraic replacement.
- Every physical bond is touched by only two two-qubit layers. Starting from
  DMRG bond dimension eight and allowing rank four per fused two-qubit gate
  gives a worst exact bond dimension of 128, with no truncation.
- `RXX`, `RYY`, and `RZZ` commute. Their product lies exactly in the four-term
  operator basis `I⊗I`, `X⊗X`, `Y⊗Y`, `Z⊗Z`; this avoids a dynamic SVD when
  applying the fused gate as a local MPO.
- The expert dispatches a compiled update 500 times from Python. A backend
  scan is semantically straightforward but may be secondary if each gradient
  contraction dominates.
- Initial parameters are near identity. Exploiting that fact to truncate small
  singular values would change the algorithm and is prohibited; exact ranks
  must be retained.

## Historical evidence, not accepted parent

`research/TASK_01_MPO_ENERGY_COMPARISON.md` reports a three-pair, different-image
experiment in which only the Hamiltonian evaluation changed and median runtime
fell from about 102.55 seconds to 67.02 seconds (1.53x). This motivates the MPO
factor but is not eligible promotion evidence and does not define the final
implementation.

## Completed measurement set

The campaign completed the immutable-reference six-run baseline, component/XLA
profiles, exact operator and 500-step trajectory checks, eleven measured
factors, and six alternating matched promotion pairs. Sanitized numbers and
per-factor figures are collected in
[`IMPLEMENTATION_COMPARISON.md`](IMPLEMENTATION_COMPARISON.md).

## Measured findings

- The reference lowering/compile path consumes about 29.4 seconds, while five
  steady updates average 43.6 ms and project to 21.8 seconds for all 500
  updates. Optimization must reduce both graph size and contraction work.
- Exact no-truncation local-MPS propagation is numerically plausible and
  reaches the predicted maximum bond dimension 128. Its expanded
  reverse-mode graph failed the 280-second compilation screen, so reducing
  tensor-network nodes at the cost of very large explicit einsums is the wrong
  end-to-end tradeoff for this workload.
- Replacing 63 Pauli-pattern measurements by one exact bond-three TFIM MPO is
  the first confirmed major factor: five candidate runs averaged 36.6112
  seconds versus 55.4067 seconds for the paired expert runs, a mean paired
  1.5124x speedup with 5/5 wins.
- A whole-training TensorCircuit `K.jaxy_scan` is not a retained factor. It was
  effectively neutral in the first three pairs and slower overall after five
  pairs (38.1729 versus 36.6960 seconds; mean paired speedup 0.9669x).
- Explicit layer-local fusion is the second major factor. It lowers the
  TensorCircuit graph from 570 primitive gate nodes to 66 fused layer-local
  nodes while preserving all 570 independent angles. Five paired runs averaged
  24.9975 seconds versus 39.0692 seconds for the MPO parent, a 1.5621x mean
  paired speedup with 5/5 wins.
- The default OMECo `16x32` search over-invests for the compact fused graph.
  OMECo `1x1` won 5/5 pairs and supplied a smaller but real third factor:
  23.8724 versus 26.5024 seconds, or 1.1039x mean paired speedup.
- More path search is not reliably better: OMECo `4x4` produced only a noisy
  1.0151x mean over `1x1` with 3/5 wins and two losses. Greedy was 17.45%
  slower in its quick screen. The retained contractor remains OMECo `1x1`.
- Batched closed-form gate assembly is the dominant factor discovered in this
  campaign. It removes hundreds of repeated scalar gate-construction
  subgraphs while still feeding exact `tc.gates.Gate` objects into the same
  TensorCircuit contraction. Five runs averaged 6.1538 seconds versus 21.9769
  seconds for native fused assembly, a 3.5755x mean paired speedup with 5/5
  wins. The full shared-input 500-step history remained within `1.57e-4`
  absolute and `3.77e-6` relative of native fused complex64 evaluation.
- Within-layer pair-product batching and whole-training scan each had a small
  positive mean but failed directional/uncertainty screening: respectively
  4/5 wins at 1.0376x and 3/5 wins at 1.0173x. Neither isolated factor is
  retained or credited in the final candidate.
- Cross-layer gate batching is a reproducible final micro-factor: it won all
  six alternating pairs over the per-layer-batched parent with a `1.0794x`
  paired mean speedup and a `1.0056x` 95% lower confidence bound.
- TensorCircuit-NG's algebraic `use_primitives=True` contractor path is not
  retained. Although its candidate runtimes were stable, only 4/6 pairs won;
  a slow parent outlier inflated the paired mean, and the confidence interval
  crossed one. This is precisely why the campaign uses paired directional and
  uncertainty gates instead of comparing pooled means alone.
- The promoted candidate beats the immutable expert in all six final matched
  pairs: `60.6511 s` versus `6.3598 s` by pooled means and `9.6364x` by mean
  pairwise speedup, with a 95% interval of `8.6808x–10.5920x`.
- The final profile makes the causal chain explicit. StableHLO size falls from
  86,824 to 2,754 lines; compilation falls from 22.63 to 2.16 seconds; steady
  updates fall from 43.60 to 5.71 ms. The MPO/fusion changes reduce
  contraction work, while batched gate assembly is the dominant graph-size
  reduction.
