# Fable 5 candidates on OrbitBreakersExpertBenchmarks (tasks 06 & 11)

Independent re-measurement of the two ORBIT-Q tasks where the Fable 5 / Cursor
agent solution appeared faster than the expert reference, using **this
repository's own paired-benchmark method** (one container per task, fresh
Python process per repeat, counterbalanced order, evaluator-reported runtime).

- Host: cloud Linux VM, 4 vCPU (bench.toml cpus/memory left unset because the
  pinned 8-CPU/9-GiB limits hit a cgroup-v2 "threaded" failure on this host).
- Method: `./bench run TASK --solution optimized --compare-to reference`.
- The `optimized` file is the Fable 5 candidate copied from
  `results/fable5/challenge-NN/solution_N.py` of the main ORBIT-Q run.
- **Precision note:** this repo's image sets `JAX_ENABLE_X64=0`, forcing single
  precision (complex64) for *both* sides — an inherently matched-precision
  harness, which removes the precision confound discussed in the main run.

## Results

| Task | Reference (s) | Fable 5 candidate (s) | Speedup | Improvement | Runs |
| ---: | ---: | ---: | ---: | ---: | --- |
| 11 | 179.59 ± 1.81 | **120.02 ± 0.71** | **1.50×** | **33.2%** | 4 ref + 4 opt, all PASS |
| 06 | unavailable* | 70.44 ± 0.42 | n/a | n/a | 6 opt PASS; reference fails |

`*` task 06: the publication reference in this repo
(`references/task-06/solution_6.py`) cannot run on the public TensorCircuit-NG
nightly — it calls `ode_evol_global(..., mode="raw")` with a hamiltonian
signature the released `timeevol.py` no longer accepts, raising
`TypeError: vf() missing 1 required positional argument: 'tt'` on both the
pinned `1.7.0.dev20260618` and the newer `1.8.0.dev20260726` images. This
matches this repository's own bootstrap note ("Task 06 reached an
installed-framework API mismatch around `ode_evol_global(..., mode='raw')`").
The Fable 5 candidate uses the public `mode="hamiltonian"` API and runs
cleanly (6/6 PASS, ~70 s), so no paired speedup can be computed against a
reference that does not execute.

## Conclusion

- **Task 11: the speedup is real and independently confirmed.** Under this
  repo's matched-precision paired protocol, the Fable 5 candidate is 1.50x
  faster than the expert reference (33.2% improvement), consistent in
  direction with the main-run matched-precision ratio (~0.74, i.e. ~1.35x
  faster). Standard errors are tight and non-overlapping.
- **Task 06: not decidable here** because the repo's reference is broken
  against public TensorCircuit-NG; the candidate itself runs correctly.

Raw per-repeat logs are under `results/fable5-c11/logs/` and
`results/fable5-c06/logs/`.
