# Task 11 Research Insights

Task: `task-11`

Last consolidated: 2026-07-28

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

none (no candidate has passed the frozen paired measurement rule yet).

## Preserved semantics

- Parameter layout (7 stacked leaves) and seeded float32 initialization
  (`default_rng(2041)`, scale 0.05).
- Layer structure: per-site spin-1 rz/ry/rz rotations, then even-bond and
  odd-bond entanglers `expm(-i[theta S.S + (phi-theta) SzSz +
  beta (S.S)^2])` with the exact generator.
- Exactly 500 sequential Adam updates at learning rate 0.03; pre-update
  energy density recorded per step.
- Energy density of `H = sum_bonds [S.S + 0.2 (S.S)^2] +
  0.15 sum_i (S_i^z)^2`; three string-order expectation values
  `<S^z_i exp(i pi sum S^z) S^z_j>`.
- Dense-state evolution and all non-diagonal observables through
  `tc.QuditCircuit`; complex64 semantics; unchanged output contract.

## Confirmed bottlenecks

- ~97% of the reference's end-to-end time is the 500-step loop (323 ms per
  step); compile is only ~4 s - the opposite regime from Task 12
  (`profiles/reference-profile.json`, ledger entry "Reference bottleneck
  profile").
- The step is memory-bandwidth-bound on the dense 3^12 complex64 state:
  one 9x9 two-qudit contraction costs 0.5-2.9 ms depending on bond position
  (transpose+gemm lowering), and the reference applies 47 gates per layer
  plus 23 expectation contractions per energy
  (`profiles/gate-application-microbench.json`).
- A bare reshape-matmul floor probe bounds layout tuning at ~1.7x below the
  framework circuit path; hand-rolled evolution is not shippable under the
  framework-fidelity rules and was not pursued.

## What worked

none recorded yet; see `SURVEY.md` hypothesis e01 for the primary planned
candidate.

## What did not work

- Layer unrolling instead of `K.scan`: 295.5 vs 264.6 ms per step in the
  external precursor; larger compile, worse scheduling.
- Splitting even/odd gates into separate circuits: 285.3 ms per step; one
  extra state materialization per layer.
- Fusing 4 sites into dim-81 blocks: 81x81 crossing gates turn the
  bandwidth-bound problem into a compute-bound one (cf. the Task 12 dim-16
  dead end). Rejected on FLOPs before implementation.

## Open hypotheses

- e01: exact gate fusion (47 -> 11 passes per layer) + batched fixed-order
  Pade(3,3) entangler exponentials + diagonal-observable coefficient vector
  + whole-training scan (see `SURVEY.md`).
- Reproduce the six-pair protocol under `--engine docker` on a
  Docker-capable host to open the formal `GOAL.md` promotion gate.

## Evidence limits

- All in-repo runtime evidence uses `--engine local` on one 4 vCPU host;
  the Docker single-container protocol has not run here.
- Only the canonical fixed workload (12 sites, 5 layers, 500 updates, seed
  2041) is measured; no scaling claim.
- Python 3.12.3 with the exact lock versions (the Docker image ships
  Python 3.11); a Docker rerun removes this deviation.
