# Local matched-environment re-measurement (tasks 06 & 11)

After the Docker paired run, both references were re-measured **locally** in a
single fresh virtualenv so that reference and candidate share exactly the same
interpreter, framework build, and precision flags. This also resolved the
Docker-only task-06 reference failure.

## Environment

- Local venv, Python 3.12, `tensorcircuit-nightly 1.8.0.dev20260726`,
  **jax 0.11.0**, latest `diffrax`, `NUMBA_DISABLE_JIT=1`, `JAX_ENABLE_X64=0`
  (single precision for both sides).
- Each evaluator invoked directly as a fresh process, 3 repeats per side.

## Results

| Task | Reference (s) | Candidate (s) | Speedup (ref/cand) | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 11 | 232.43 ± 10.83 | **128.32 ± 3.13** | **1.81×** | candidate faster — speedup confirmed |
| 06 | **71.83 ± 0.09** | 79.39 ± 2.28 | 0.905× | **reference faster by ~10.5%** |

## Two corrections to earlier claims

1. **task-06 reference is NOT fundamentally broken.** It fails in the pinned
   Docker image only because that image ships `jax 0.10.0`, whose diffrax
   integration is incompatible with tensorcircuit's `ode_evol_global` diffrax
   path (`TypeError: vf() missing 'tt'`). With `jax 0.11.0` + current diffrax
   the reference runs cleanly (~72 s). The earlier "reference unavailable on
   public TC" note applies only to the jax-0.10.0 pin, not to the algorithm.

2. **task-06 is NOT a candidate speedup.** In this matched local environment
   the expert reference (71.8 s) is ~10.5% faster than the Fable 5 candidate
   (79.4 s). The main run's "matched-precision 0.76" figure for task-06 was an
   artifact of comparing a complex64 candidate variant against a reference
   runtime that had not been re-measured under the same precision; it does not
   survive a genuinely matched comparison and should be treated as retracted.

## Net conclusion for the two "faster than expert" tasks

- **task-11: the speedup is real and reproduced twice** (Docker paired 1.50×,
  local matched 1.81×; both single precision, tight non-overlapping spreads).
- **task-06: the earlier speedup does not hold** once the reference is measured
  in the same environment; the expert reference is modestly faster.

So of the two candidates flagged as faster than expert, **only task-11 is a
genuine artifact-efficiency win**; task-06 was a measurement artifact and is
corrected here.
