# Task 07 Research Insights

Task: `task-07`

Last consolidated: 2026-07-29

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment `e11` analytically eliminates the measured ancilla subsystem,
deduplicates the 64 fixed trajectories into two weighted patterns, and runs
the remaining eight-qubit data circuits with TensorCircuit's native local
contractor. Six final counterbalanced Docker pairs all pass and all win:
candidate mean 3.070839 seconds versus expert mean 140.076441 seconds; mean
paired speedup 45.757921x (95% Student-t CI
39.384711x-52.131131x).

This is explicitly a challenge-design reduction. The conservative `e04a`
implementation remains available when literal 16-qubit `cond_measure`
execution is required; its six-pair result is 4.478752x.

## Preserved semantics

- Two adaptive layers and all 96 float32 parameters in the expert's layout;
  ancilla rotation parameters remain in place but have their exact zero
  pathwise gradients.
- Seed 2047 parameter initialization and seed 2048 fixed trajectory uniforms.
- The exact measured/source bits selected by all 1,024 fixed-uniform
  comparisons, with the selected trainable feedback branch for every bit.
- Exactly 64 fixed trajectories averaged per objective and exactly 100
  sequential Adam updates at learning rate 0.02; equal trajectories are
  evaluated once with exact multiplicity weights.
- Pre-update energy history and post-update per-trajectory energy vector.
- The eight-site open-boundary TFIM Hamiltonian and complex64 TensorCircuit
  quantum computation on the remaining data register.

Not preserved literally: construction of the eight ancilla qubits and
framework-native `cond_measure` calls. That distinction is the loophole and
must remain visible in any PR.

## Confirmed bottlenecks

- Every trajectory evaluates seven `ZZ` and eight `X` expectations separately,
  repeating the final circuit's bra/ket contraction 15 times in both forward
  and reverse-mode work.
- Approximately 48 seconds is fixed trace/compile/path/finalization cost; the
  additional 99 canonical updates average about 0.89 seconds each.
- The generic `conditional_gate` keeps a selected dense two-qubit `RZZ` node
  after the ancilla is already a Z eigenstate.

## What worked

- The ancilla circuit is exactly a classical Bernoulli source followed by a
  prefix-XOR permutation. Full TensorCircuit and analytic sampling agree on
  all 1,024 measured bits.
- The fixed public batch has only two unique complete patterns, with counts
  63 and 1. Replacing 64 sixteen-qubit trajectory graphs by two weighted
  eight-qubit circuits lowers the canonical screen to about 3 seconds.
- TensorCircuit's `plain-experimental` local contractor is better suited to
  the reduced graph than greedy or OMECo-1x1. It beats greedy in 6/6
  contractor pairs.
- One TensorCircuit state contraction plus one native sparse eight-qubit TFIM
  expectation reduces the 100-step screen from 135.816 to 61.397 seconds and
  the 50-step passing screen from 91.540 to 52.769 seconds.
- One-trajectory energy and gradient agree within `3.34e-6` and `1.01e-6`;
  the full 50/100-step physical outputs pass and remain close.
- Reducing post-measurement
  `RZZ(theta_b)|b,psi>` to
  `|b> RZ((1-2b)theta_b)|psi>` removes 16 selected two-qubit nodes. The
  identity audit's maximum complex64 error is `2.98e-8`; canonical runtime
  falls again from 61.397 to 33.546 seconds.

## Measurement lesson

- Maximum first-Adam parameter difference is a poor complex64 equivalence
  gate near zero gradients. The strict e01 diagnostic failed (`0.0313`) even
  though energy, gradient, post-update energy, and complete public workloads
  passed. Preserve that failure, but use physical post-update outputs as the
  predeclared one-step semantic criterion in later experiments.

## What did not work

- On the reduced graph, whole-training scan remains slower (3.206 versus
  2.998 seconds in the exploratory screen), and explicit `RZ*RY` dense-gate
  fusion is slower again at 3.531 seconds.
- Whole-training `K.jaxy_scan` is correct but slower after e02: 36.747 versus
  33.546 seconds for 100 steps and 34.917 versus 31.300 seconds for 50.
  Control-flow compilation outweighs only 100 cached-JIT host dispatches.
- TensorNetwork greedy takes 23.234 seconds for the frozen one-step screen,
  12.39% slower than OMECo 1x1, so it was discarded before full training.
- Joint TensorCircuit-state measurement rounds are the fastest 1/50-step
  method (4.606/14.523 seconds) and closely reproduce the expert, but dense
  16-qubit state differentiation raises the 100-step time to 26.531 seconds,
  8.90% slower than e04a. This exposes a crossover between staging cost and
  per-update dense-state cost.
- TensorCircuit `K.vvag` is 90.69% slower at one step (39.420 seconds) because
  it maps individual reverse-mode programs; differentiating the shared mapped
  mean remains superior here.

## Contractor result

- After simplifying the graph, OMECo 32x32 over-searches. The 1x1 budget
  lowers the passing canonical screen from 33.546 to 24.362 seconds and the
  50-step screen from 31.300 to 21.473 seconds without hurting convergence.

## High-confidence exact identities

- Final ancillas factor from the data state because the last operation that
  can mix their computational basis is followed by `cond_measure`, and the
  remaining feedback is diagonal. This permits one full TensorCircuit state
  contraction followed by an eight-qubit native Hamiltonian expectation.
- On a measured ancilla `|b>`, feedback `RZZ(theta_b)` is exactly data
  `RZ((1-2b) theta_b)` with the ancilla unchanged.
- A `K.jaxy_scan` can emit the same pre-update values while carrying the same
  Optax state and final parameters.

## Open hypotheses

- None for the current fixed workload. Further closed-form elimination of the
  eight-qubit data circuit would likely violate the framework-fidelity policy
  and is unnecessary for exposing the challenge-design issue.

## Evidence limits

- No matched external implementation exists, so “SOTA” can mean only the
  campaign-best implementation for this repository workload.
- No scaling or cross-hardware claim is supported.
- The 45.76x implementation satisfies the executable output contract but may
  be rejected if maintainers interpret Task 07 as requiring literal
  mid-circuit TensorCircuit measurement. The conservative 4.48x candidate is
  the fallback under that interpretation.
- In the earlier conservative e04a run, long-session thermal/system noise
  widened candidate times to 24.222-31.684 seconds; no value was filtered.
  The final e11 candidate ranged from 2.967 to 3.312 seconds, also without
  filtering.
