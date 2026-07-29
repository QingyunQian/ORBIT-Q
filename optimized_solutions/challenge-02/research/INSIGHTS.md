# Task 02 Research Insights

Task: `task-02`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

The accepted implementation combines whole-training TensorCircuit
`K.jaxy_scan` with batched checkpoint states and exact Frobenius purity.
Against the scan-only parent, the entropy combination achieved `1.074457x`
mean paired speedup, 95% interval `[1.044174x, 1.104740x]`, and `6/6` wins.
The final immutable-reference comparison is recorded separately after source
freeze.

## Preserved semantics

- 12-qubit Neel input and complex64 TensorCircuit-NG evolution;
- six brickwork sublayers with all 243 independent parameters;
- three half-chain Renyi-2 checkpoints and the fixed target profile;
- the 45-term open XXZ plus staggered-field Hamiltonian;
- fixed seed 2026, Adam learning rate `0.015`, and 500 sequential updates;
- every pre-update history and the original four-key NumPy result contract.

## Confirmed bottlenecks

- One update lowers in `0.803158 s`, compiles in `1.929769 s`, and contains
  15,479 StableHLO lines. The 500 steady updates project to `1.321653 s`, so
  both cold graph staging and steady execution are material.
- The separately compiled trajectory is about 93.5% of the forward loss
  runtime (`0.000889/0.000951 s`), making exact gate-graph reduction the
  highest-priority factor.
- The host dispatches one compiled optimizer update 500 times. A whole-training
  scan can remove that boundary, although it cannot eliminate the steady
  quantum kernel itself.
- The termwise Hamiltonian MVP traces 45 independent paths, but its separately
  measured forward action is only `0.000176 s`. Sparse conversion is therefore
  a secondary compile-size hypothesis, not the primary steady bottleneck.
- One generic Renyi-2 call is `0.000132 s`; its Frobenius-purity rewrite is
  algebraically exact but expected to be a smaller factor than gate fusion.

Profile: [`profiles/reference-profile.json`](profiles/reference-profile.json).

## What worked

- Whole-training `K.jaxy_scan` preserved the four 12-step histories
  bit-for-bit and reduced canonical mean runtime from `4.724217` to
  `4.563051 s` in its six-pair session.
- Returning all three block checkpoints and batching the exact
  `-log(sum(rho * conj(rho)))` calculation with `K.vmap` produced a further
  `1.074457x` paired gain over scan-only, with its confidence interval wholly
  above one.
- The combination result is not attributed to either subfactor alone:
  `K.vmap` alone (`1.046783x`) and Frobenius purity alone (`1.024675x`) both
  had wide intervals crossing one. The confirmed unit is their shared kernel.

## What did not work

- Exact local gate fusion reduced the gate-application count but regressed to
  `0.965849x`; dynamic trigonometry and matrix assembly cost more than the
  saved applications.
- TensorCircuit sparse XXZ action regressed to `0.972882x`; the 45-term
  termwise MVP is preferable at this size.
- Packing the ten-leaf PyTree into one tensor was bit-exact but collapsed to
  `0.428014x` due to slice/gather/scatter compile and autodiff overhead.
- Frobenius purity alone and checkpoint `K.vmap` alone were inconclusive and
  are not claimed independently.
- Direct batched pure-state Gram matrices were rejected before timing: a
  different complex64 contraction order exceeded the frozen trajectory
  tolerance despite algebraic equivalence.

## Open hypotheses

1. larger-qubit scaling, which is outside this frozen workload;
2. future TensorCircuit compiler changes that may alter the discarded
   gate-fusion or sparse-kernel tradeoffs.

## Evidence limits

Evidence covers only the canonical public Task 02 configuration, one latest
TensorCircuit-NG image, and one 6-CPU/7-GiB same-host profile. It does not
establish global SOTA or scaling. End-to-end differences include cold import,
tracing, compilation, 500 updates, synchronization, and conversion.

The final six-pair comparison records `1.115649x` mean paired speedup with a
95% Student-t interval `[1.057686x, 1.173613x]` and `6/6` wins.
