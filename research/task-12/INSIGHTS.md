# Task 12 Research Insights

Task: `task-12`

Last consolidated: 2026-07-28

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment `e01` (`src/solutions/task-12/solution_12.py`, SHA-256
`1d9a36c2649e938666f680e98e423e966f82c503da64ce7e4e06c5e33344560a`): batched
fixed-order su(4) exponentials plus a whole-training `jax.lax.scan`. Six
eligible local-engine pairs against the immutable reference: paired speedup
mean 3.914x ± 0.015x (95% Student-t CI 3.877x-3.951x), candidate mean
2.320613 s vs reference mean 9.082742 s, 6/6 pairs won. The tracked
pair-fused variant (`variants/solution_12_fused.py`) reaches 4.248x ± 0.020x
(CI 4.196x-4.299x) under the same protocol. Scope: this repository's Task 12
canonical workload on the campaign host with `--engine local`; the Docker
promotion gate of `GOAL.md` remains closed pending a Docker rerun.

## Preserved semantics

- su(4) parameterization of every SU4 gate in the exact
  `tc.gates.su4_gate` generator order; gates enter the circuit as unitaries
  `expm(-1j * sum_i theta_i G_i)`.
- Seeded float32 initialization (`default_rng(2039)`, scale 0.02), exactly
  5000 sequential Adam updates at learning rate 0.02, pre-update loss and
  fidelity recorded per step.
- Direct tensor-network overlap of the evaluator-provided DMRG-MPS bra with
  the circuit ket; no target-state preparation circuit, no oracle values.
- complex64 TensorCircuit/JAX semantics; full output contract
  (`loss_history`, `fidelity_history`, `final_parameters`,
  `final_overlap_phase`).

## Confirmed bottlenecks

- ~60% of the reference's end-to-end time is jit trace plus XLA compile of an
  8884-line StableHLO module; the 31 separate `su4` constructions (15
  scalar multiply/stacks and one norm-adaptive Pade-13 `expm` per gate,
  all differentiated) dominate the graph
  (`profiles/reference-profile.json`, ledger entry "Reference bottleneck
  profile").
- Inside the step, gate construction dominates: batched-gate-build-only scan
  steps cost 0.381 ms (adaptive expm) vs 0.123 ms (fixed Pade), while the
  whole candidate step is 0.190 ms; the tensor-network contraction is nearly
  free because the exact circuit state has bond dimension <= 4
  (`profiles/expm-microbench.json`, `profiles/reference-profile.json`).
- Per-step Python dispatch is only ~5 us, so a scan alone is a single-digit
  percentage improvement (external precedent, recorded in `SURVEY.md`
  hypothesis 3).

## What worked

- One einsum against the stacked 15 su(4) generators plus one batched
  fixed 2^5 scaling-and-squaring diagonal Pade(3,3) exponential for all 31
  gates: exactly unitary for anti-Hermitian input (measured unitarity defect
  <= 8.3e-6 in complex64), max deviation from a float64 exponential 3.2e-6
  over generator norms up to 33.6 while the training trajectory never
  exceeds norm 3.61 (`profiles/equivalence-check.json`); shrinks StableHLO
  from 8884 to 954 lines, compile from ~3.4 s to ~0.4 s, and the step from
  0.667 ms to 0.190 ms.
- `jax.lax.scan` over the 5000 Adam updates: compiles the step exactly once
  and removes per-step dispatch; essential once the step itself is ~0.2 ms.
- Pair-fusing qubit pairs into 16 four-level `QuditCircuit` sites (tracked
  variant): halves the contraction network; worth a further ~8% end to end
  after the gate-construction fix (4.248x vs 3.914x).

## What did not work

- Scan alone on the unmodified reference objective: ~9% end to end
  (external precedent; dispatch is not the bottleneck). Do not resubmit as a
  standalone candidate.
- Fusing four qubits per site (dim-16 sites, 256x256 crossing gates):
  contraction FLOPs explode; measured ~4x slower than the reference
  formulation in the external precursor. Do not repeat unchanged.
- Absorbing the 16 single-site fused gates into the 15 two-site gates:
  the absorption einsums cost what the removed network nodes saved.
- `lax.scan` with `unroll=4`: larger compile and slower steps.
- XLA CPU thread flags and core pinning: no gain; the kernels are too small
  to parallelize.

## Open hypotheses

- Reproduce the six-pair protocol under `--engine docker` on a Docker-capable
  host to open the formal `GOAL.md` promotion gate (highest value; no code
  change required).
- A hand-scheduled 16-block transfer-matrix contraction inside the scan body
  could cut the remaining ~0.06 ms of network kernels per step, but departs
  from framework circuit APIs and was deliberately not pursued; revisit only
  with maintainer guidance on framework-fidelity boundaries.
- Trimming the residual ~1.15 s of trace+compile (e.g., a lighter contractor
  search for the 63-node fused network) is bounded by Amdahl to <25% of the
  candidate's remaining runtime.

## Evidence limits

- All in-repo runtime evidence uses `--engine local` on one 4 vCPU host;
  the Docker single-container protocol has not run here.
- The campaign measured only the canonical fixed workload (32 qubits, 2
  layers, 5000 updates, seed 2039); no scaling claim is made.
- Python 3.12.3 was used with the pinned lock (the Docker image ships
  Python 3.11); package versions match the lock exactly.
