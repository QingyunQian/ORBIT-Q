# Reproduction: Fable 5 tasks 06 & 11 on OrbitBreakersExpertBenchmarks

This folder preserves an independent re-measurement of the two ORBIT-Q tasks
where the Fable 5 / Cursor-agent candidate looked faster than the expert
reference, run under **hmyuuu/OrbitBreakersExpertBenchmarks**'s own paired
benchmark method. That upstream repo is not writable by this agent, so the
work is bundled here for retrieval.

## Headline result

Two measurement rounds — Docker paired (repo method) and a local matched-env
cross-check. **Only task-11 is a genuine speedup; task-06 is not.**

| Task | Round | Reference (s) | Candidate (s) | Speedup | Verdict |
| ---: | --- | ---: | ---: | ---: | --- |
| 11 | Docker | 179.59 ± 1.81 | 120.02 ± 0.71 | **1.50×** | candidate faster |
| 11 | local | 232.43 ± 10.83 | 128.32 ± 3.13 | **1.81×** | candidate faster |
| 06 | Docker | unavailable* | 70.44 ± 0.42 | n/a | reference fails on jax-0.10.0 pin |
| 06 | local | 71.83 ± 0.09 | 79.39 ± 2.28 | 0.905× | **reference faster ~10.5%** |

`*` the task-06 reference is not fundamentally broken; it fails only under the
image's pinned `jax 0.10.0`. With `jax 0.11.0` + diffrax it runs (~72 s), and
then it is *faster* than the candidate. See `LOCAL_RESULTS.md` for the full
correction. The repo image sets `JAX_ENABLE_X64=0`, so both sides are complex64
(matched precision) in both rounds.

## Contents

- `REPORT.md` — full method, results, and caveats.
- `fable5-c06-c11.bundle` — a git bundle of the branch
  `cursor/fable5-c06-c11-benchmark-7148` created against the upstream repo,
  containing: the two candidate solutions staged into `src/solutions/`, a
  registered `tensorcircuit-py311-nightly0726` environment, and the raw
  per-repeat measurement logs under `results/`.

## How to push this to the upstream repo (needs your GitHub account)

```bash
# 1. Fork hmyuuu/OrbitBreakersExpertBenchmarks on GitHub to your account.
# 2. Clone your fork and import the bundle:
git clone https://github.com/<you>/OrbitBreakersExpertBenchmarks.git
cd OrbitBreakersExpertBenchmarks
git fetch /path/to/fable5-c06-c11.bundle cursor/fable5-c06-c11-benchmark-7148
git checkout -b cursor/fable5-c06-c11-benchmark FETCH_HEAD
git push -u origin cursor/fable5-c06-c11-benchmark
# 3. Open a PR from your fork branch to hmyuuu/OrbitBreakersExpertBenchmarks.
```
