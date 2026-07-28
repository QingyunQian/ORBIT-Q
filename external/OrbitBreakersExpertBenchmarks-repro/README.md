# Reproduction: Fable 5 tasks 06 & 11 on OrbitBreakersExpertBenchmarks

This folder preserves an independent re-measurement of the two ORBIT-Q tasks
where the Fable 5 / Cursor-agent candidate looked faster than the expert
reference, run under **hmyuuu/OrbitBreakersExpertBenchmarks**'s own paired
benchmark method. That upstream repo is not writable by this agent, so the
work is bundled here for retrieval.

## Headline result

| Task | Reference (s) | Fable 5 candidate (s) | Speedup | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 11 | 179.59 ± 1.81 | 120.02 ± 0.71 | **1.50×** | speedup confirmed (matched precision) |
| 06 | unavailable | 70.44 ± 0.42 | n/a | repo reference broken on public TC |

The repo image sets `JAX_ENABLE_X64=0`, so both sides run complex64 — an
inherently matched-precision comparison. Full write-up in `REPORT.md`.

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
