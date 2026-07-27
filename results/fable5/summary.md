# Fable 5 / TensorCircuit-NG Run Summary

| Task | Solved | Functional | Static | Runtime (s) | Codex audit | Reward | Notes |
| ---: | :---: | :---: | :---: | ---: | :---: | :---: | --- |
| 01 | yes | 1.0 | 1.0 | 178.5 | 1.0 (gpt-5.6-sol) | **1.0** | official stamp 2026-07-27; cloud precheck runtime 150.4s |
| 02 | yes | 1.0 | 1.0 | 16.7 | pending | pending | cloud precheck passed; awaiting official stamp |
| 03 | - | - | - | - | - | - | |
| 04 | - | - | - | - | - | - | |
| 05 | - | - | - | - | - | - | |
| 06 | - | - | - | - | - | - | |
| 07 | - | - | - | - | - | - | |
| 08 | - | - | - | - | - | - | |
| 09 | - | - | - | - | - | - | |
| 10 | - | - | - | - | - | - | |
| 11 | - | - | - | - | - | - | |
| 12 | - | - | - | - | - | - | |

Runtime is the evaluator's timed `run_solution(config)` wall time from the
cloud precheck (4 vCPU x86_64); the official stamp records its own runtime in
`challenge-NN/reward.json`.

Challenge-01 physics summary (cloud precheck): DMRG reference -41.50400741,
initial variational energy identical to reference (diff 1.6e-13), final energy
within 1.3e-6 of reference after the fixed 500-step Adam refinement; all five
evaluator criteria PASS.

Challenge-02 physics summary (cloud precheck): exact sparse GS energy density
-2.00036788; energy density optimized from -0.7412 to -1.9925 (gap 0.0079 to
exact GS, criterion allows 1.0); loss decreased -0.6538 -> -1.9683; final
block entropies [0.305, 0.594, 0.261] vs targets [0.30, 0.60, 0.80] (MSE
0.0968) - the third checkpoint stays low because the energy term dominates
the 0.25-weighted entropy penalty near the ground state, which is the honest
optimum of the prescribed loss; all evaluator criteria PASS.
