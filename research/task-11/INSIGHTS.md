# Task 11 Research Insights

Task: `task-11`

Last consolidated: 2026-07-28

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment `e01` (`src/solutions/task-11/solution_11.py`, SHA-256
`a787e7579bbe969f25e3ca9bea0b7f25b028e6e0bd7f8d54a6969940de1149c5`): exact
gate fusion (47 -> 11 dense-state passes per layer), batched fixed-order
entangler exponentials, diagonal onsite coefficient vector, and a
whole-training `jax.lax.scan`. Six eligible local-engine pairs against the
immutable reference: paired speedup mean 1.464x ± 0.003x (95% Student-t CI
1.457x-1.472x), candidate mean 114.968325 s vs reference mean 168.361539 s,
6/6 pairs won. Scope: this repository's Task 11 canonical workload on the
campaign host with `--engine local`; the Docker promotion gate of `GOAL.md`
remains closed pending a Docker rerun.

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

- Exact gate fusion: compose the three single-site rotations per site into
  one 3x3 unitary and absorb each even-bond pair into its 9x9 entangler so
  every layer applies 11 two-qudit unitaries instead of 47 gate applications
  (`profiles/reference-profile.json`: StableHLO 8135 -> 2426 lines; step
  323 -> 268 ms; forward energy 36 -> 17.5 ms).
- Batched fixed 2^5 scaling-and-squaring diagonal Pade(3,3) for all
  entanglers of a layer: exactly unitary for anti-Hermitian input; max
  generator norm over training is 2.959 (`profiles/equivalence-check.json`).
- Precomputed per-basis-state coefficient vector for the diagonal single-ion
  term: replaces 12 separate `expectation` contractions with one weighted
  `|amp|^2` sum.
- `jax.lax.scan` over the 500 Adam updates plus a jitted finalize block:
  compiles the step once and removes per-step Python dispatch.

## What did not work

- Layer unrolling instead of `K.scan`: 295.5 vs 264.6 ms per step in the
  external precursor; larger compile, worse scheduling.
- Splitting even/odd gates into separate circuits: 285.3 ms per step; one
  extra state materialization per layer.
- Fusing 4 sites into dim-81 blocks: 81x81 crossing gates turn the
  bandwidth-bound problem into a compute-bound one (cf. the Task 12 dim-16
  dead end). Rejected on FLOPs before implementation.

## Open hypotheses

- Reproduce the six-pair protocol under `--engine docker` on a
  Docker-capable host to open the formal `GOAL.md` promotion gate (highest
  value; no code change required).
- Further dense-state layout tuning inside `QuditCircuit` is bounded by the
  ~1.7x reshape-matmul floor probe and would depart from framework circuit
  APIs; revisit only with maintainer guidance on framework-fidelity
  boundaries.

## Evidence limits

- All in-repo runtime evidence uses `--engine local` on one 4 vCPU host;
  the Docker single-container protocol has not run here.
- Only the canonical fixed workload (12 sites, 5 layers, 500 updates, seed
  2041) is measured; no scaling claim.
- Python 3.12.3 with the exact lock versions (the Docker image ships
  Python 3.11); a Docker rerun removes this deviation.
