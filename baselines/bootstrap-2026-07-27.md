# Shared-container bootstrap results

**Date:** 2026-07-27  
**Status:** bootstrap evidence; not a promotion or SOTA claim  
**Host allocation:** 8 CPUs, 9 GiB Docker limit  
**Method:** two counterbalanced pairs per task in one container per task; every
cell is a fresh evaluator process.

The `optimized` files were byte-identical to their immutable references during
this run. Therefore, nonzero percentages below measure run/order noise, not
algorithmic improvement.

| Task | Reference mean ± SEM (s) | Optimized mean ± SEM (s) | Paired improvement mean ± SEM | Outcome |
| --- | ---: | ---: | ---: | --- |
| 01 | 59.596 ± 0.325 | 59.344 ± 0.319 | 0.417% ± 1.080% | 2/2 pairs passed |
| 02 | 5.730 ± 0.114 | 5.828 ± 0.088 | -1.786% ± 3.565% | 2/2 pairs passed |
| 03 | 5.037 ± 0.116 | 5.143 ± 0.055 | -2.189% ± 3.455% | 2/2 pairs passed |
| 04 | 18.637 ± 0.296 | 19.261 ± 1.029 | -3.463% ± 7.160% | 2/2 pairs passed |
| 05 | 116.277 ± 3.728 | 115.078 ± 2.944 | 1.011% ± 0.642% | 2/2 pairs passed |
| 06 | unavailable | unavailable | unavailable | symmetric nonzero exit |
| 07 | 109.332 ± 0.081 | 124.904 ± 5.291 | -14.247% ± 4.924% | 2/2 pairs passed |
| 08 | unavailable | unavailable | unavailable | symmetric resource failure |
| 09 | 33.089 ± 0.282 | 32.159 ± 0.023 | 2.804% ± 0.760% | 2/2 pairs passed |
| 10 | 19.996 ± 0.930 | 20.122 ± 0.028 | -0.857% ± 4.835% | 2/2 pairs passed |
| 11 | 153.612 ± 5.039 | 153.159 ± 4.542 | 0.284% ± 0.314% | 2/2 pairs passed |
| 12 | 11.261 ± 0.972 | 10.935 ± 0.092 | 2.093% ± 9.260% | 2/2 pairs passed |

Challenge 06 reached an installed-framework API mismatch around
`ode_evol_global(..., mode="raw")`. Challenge 08 exceeded the pinned container
memory while evaluating its monolithic sampling vectorization. Both roles had
the same source bytes and the same terminal class; the benchmark correctly
reported no timing or improvement for those tasks.

An additional four-pair runner validation on Challenge 02 measured:

| Reference mean ± SEM (s) | Optimized mean ± SEM (s) | Paired improvement mean ± SEM |
| ---: | ---: | ---: |
| 5.834799 ± 0.043621 | 5.830682 ± 0.101475 | 0.055216% ± 1.880322% |

## Evidence

- Full bootstrap report:
  `results/all-shared-r2-bootstrap-v2/results.json`
  (`sha256:c2fc02407e8d14c9e9858abed7ddcca5ecb8260e638947507dbda7b4032e0d40`)
- Full bootstrap summary:
  `results/all-shared-r2-bootstrap-v2/summary.json`
  (`sha256:7c80b627a6a367d2883a2e40f8e64af44f1d065fc756bd27abaa730f311ee0e7`)
- Challenge 02 four-pair report:
  `results/task-02-shared-r4/results.json`
  (`sha256:3bc8c9807fff53e32b53855c2d00ee307abdb087140ee192767d383dd8556932`)
- Challenge 02 four-pair summary:
  `results/task-02-shared-r4/summary.json`
  (`sha256:5ac70898f7ca2c29a5fd723816d293538b2df8f63663885aa796cc861b286327`)

The report records one actual container ID and one staging-snapshot hash per
task, alternating `reference → optimized` / `optimized → reference` order,
the container launch command with CPU and memory limits, the image ID, source
and evaluator hashes, and every terminal result.
