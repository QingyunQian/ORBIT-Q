# Task 03 Research Insights

Task: `task-03`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

The exact product-state candidate is the campaign best. Six final pairs passed
12/12 cells and produced `4.435277x` mean paired speedup with a 95% t-interval
of `[4.287006x, 4.583549x]`. The expert/candidate means were
`4.101350 / 0.924608 s`.

## Preserved semantics

- Twelve logical qubits and ten alternating brickwork layers.
- Every independent `RXX`, `RZZ`, and `RX` parameter with seed 2027.
- Sixty zero-outcome post-selection events and their conditional
  probabilities.
- Exact open TFIM energy, probability-aware loss, 300 Adam updates, and all
  pre-update histories.
- Complex64 TensorCircuit-NG computation and the original NumPy contract.

## Confirmed bottlenecks

The expert traces ten 12-qubit circuits, 60 projector insertions, and a 23-term
Hamiltonian MVP. Its compiled update has 10,054 StableHLO lines, with
`0.659 s` lowering and `1.656 s` compilation. Compiled updates average
`5.57 ms`, so both cold compilation and 300 host dispatches are material.

The final exact graph has 2,870 StableHLO lines, lowers in `0.249 s`, compiles
in `0.593 s`, and executes all 300 updates in a `28.9 ms` compiled scan.

## What worked

- Exact product-state reduction: `1.659987x` paired over the expert.
- `K.vmap` over local maps: `2.062555x` over scalar local maps; the dominant
  single factor.
- Whole-training `K.jaxy_scan` after graph reduction: `1.149293x`.
- `K.vmap` over six X/Z expectations: `1.092866x`.

Every retained factor won 6/6 pairs and its 95% paired-speedup lower bound
exceeded one.

## What did not work

Python-unrolling five product-state blocks passed but regressed to `7.198 s`;
retain the block-level scan already used by the expert.

Whole-training scan before local graph compression regressed to `8.990 s`.
The same factor became positive only after block scan and `K.vmap`, so this is
an interaction-sensitive optimization.

The exact Pauli/Schmidt conditional map screened at `0.981 s` versus
`0.969 s` for direct TensorCircuit gate matrices and was discarded.

## Open hypotheses

No untested hypothesis has enough expected value to justify invalidating the
final six-pair record. The simultaneous full-state projector is dominated by
the exact product reduction. Further handwritten gate algebra would lower
framework clarity and already failed its first exact Schmidt screen.

## Evidence limits

The campaign covers one fixed public workload, one image, and one host resource
profile. It establishes neither cross-hardware performance nor global SOTA.
Complex64 contraction reassociation changes the sign of a theoretical
order-`1e-9` zero gradient, which Adam magnifies in one redundant parameter;
post-update physical values and histories remain equivalent below `4.77e-7`.
