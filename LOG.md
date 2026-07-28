# Task 05 Autoresearch Campaign

Task: `task-05`

Campaign branch: `codex/orbitbreakers/task-05/gate-7f3c2d`

Campaign task: `task-05`

Live upstream PR list inspected at: `2026-07-27T18:15:35Z`

Open PRs observed:

- `sxzgroup/ORBIT-Q#3`: repository-wide reward aggregation policy fix.
- `sxzgroup/ORBIT-Q#2`: ForgeCode agent integration.

No open improvement, optimization, performance, or runtime PR targets Task 05.
Every hypothesis worktree in this campaign must remain on `task-05`.

## Campaign objective

Reduce evaluator-reported runtime for the immutable Task 05 human-expert
solution while preserving all functional checks, ten non-unitary cooling
layers, per-layer normalization, differentiation through normalization, and
exactly 600 Adam updates. The stretch target is a valid 10x paired speedup.

## Gate status before candidate work

Recorded at `2026-07-27T18:15:35Z` from parent commit
`46d6636881500fa8f70618b74f89353a2b6702b4`.

- Survey: closed; the repository scaffold is not Task 05 complete or `READY`.
- Public workload dataset: closed; the manifest is `not_built`.
- Trusted controller: closed; no external sanitized attestation was supplied.
- Repeated reference promotion gate: closed; no current six-run Task 05
  baseline report was supplied.

The candidate and immutable reference both have SHA-256
`ccafe626865ee39b651adaeead86b8bf6f541e3f1426da4842da92b6a0ee015f`.
No candidate code may change until all three knowledge/data/isolation gates
pass. Repeated reference evidence may be collected before that point, but no
speedup claim may be made from historical or unmatched numbers.

## Permitted data

- Task 05 problem, evaluator, immutable expert, and tracked expert-derived
  OMECo variant.
- Public repository/framework sources and documentation.
- Public workload records and aggregate benchmark reports.

No hidden tuning or holdout data, identifiers, paths, seeds, credentials,
per-case output, or controller logs may enter this worktree.

## Append-only campaign events

- `2026-07-27T18:15:35Z`: selected `task-05`; inspected the live upstream PR
  list; recorded all gates as closed; began public gate preparation only.

## Append-only corrections

Append corrections below this heading. Never rewrite an earlier result after it
has informed an experiment.

- `2026-07-27T18:18:21Z`: canonical public-workload validation completed.
  Immutable expert status `SUCCESS`, `Overall: PASS`; evaluator runtime
  `135.503222 s`; initial/final/exact energy densities
  `-1.1720402241`, `-1.3267312050`, and `-1.3268985748`. Report SHA-256
  `b5defee28534cb68cb274563a4f8c1075acc38ed2d1b6e8cb13acf401e8011b4`.
  This one run validates the public record but is not a performance baseline.

- `2026-07-28T01:16:00Z`: six-run immutable-reference promotion baseline
  completed in one pinned Docker container. All six runs passed. Runtimes:
  `101.547164`, `106.278911`, `114.090078`, `139.061275`, `122.631583`,
  `123.046707` seconds. Mean `117.775953 s`; median `118.3608305 s`;
  sample standard deviation `13.5171310916 s`; standard error
  `5.5183456601 s`; min/max `101.547164/139.061275 s`. Host fingerprint
  `d72d96a55e39ff10c67a820a30902dbd1b919a8f41fb4dbf95c855eac59f0013`;
  image ID
  `sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c`.
  Report SHA-256
  `529a1839c67c55bece0b89b82ffd3583868a082192c2f357a819566ac1463b76`.
  The repeated-reference gate passes for Task 05. No improvement is claimed
  without matched candidate pairs.

- Correction recorded `2026-07-28T01:19:00Z`: the preceding baseline event's
  authoritative completion time is `2026-07-28T01:18:25.118354Z`, not
  `2026-07-28T01:16:00Z`.

- `2026-07-28T01:24:06Z`: immutable-reference profiling completed after
  preserving two failed profiler attempts. The first attempt called an
  ahead-of-time executable after the MVP closure's mutable cache changed its
  signature (`compiled for 29 inputs but called with 7`). The second reused
  one `PauliStringSum2MVP` closure across independent JAX traces and produced
  `UnexpectedTracerError`. The successful profiler used a fresh public MVP
  closure per independent trace and did not modify the expert.

- `2026-07-28T01:24:06Z`: successful profile report
  `research/profiles/task-05-reference-profile.json`, SHA-256
  `812469668c2a571cbf0119ee30f041a225a1d4721b89ddd34ded5210b727ad67`.
  Eight steady update executions averaged `0.1818234792 s`, projecting to
  `109.0940875 s` for 600 updates. XLA reported about `209,987,840` FLOPs,
  `1,187,454,208` bytes accessed, and `289,466,256` temporary bytes per update.
  Update lowering/compilation took `0.637576/1.324448 s`. Interpretation:
  steady TensorCircuit gradient execution, not compilation or Python dispatch,
  is the primary 10x barrier.

- Correction recorded `2026-07-28T01:26:02Z`: the profiler was rerun after
  finalizing its output filter so the executed script bytes match profiler
  SHA-256
  `921c302a46e2d394022a658c574c62f8c3adb7913019531932aea4a164895f91`.
  The authoritative report is still
  `research/profiles/task-05-reference-profile.json`, now SHA-256
  `be24858b7693ff10c1c153a7fb27ba73a2b60fa7eae5e74ea16be9aa74e6473c`.
  Its steady update mean is `0.1697164166 s`, projected 600-update execution
  `101.82984997 s`, and lowering/compilation times
  `0.5897331670/1.2868244170 s`. The XLA operation/traffic estimates and
  interpretation are unchanged.

- `2026-07-28T01:28:18Z`: immutable forward-component profile completed.
  Report `research/profiles/task-05-component-profile.json`, SHA-256
  `9096f74abc7d1f3b3a9ba902f70e467c530020a050194bc05b72e3392e482bee`;
  profiler SHA-256
  `a3d902878d1fd95c5d80869a0b1f36a5ac7ea94a842448ba68a4719e3b39f4b1`.
  The separately compiled ten-layer trajectory and 35-term Hamiltonian energy
  had steady medians `0.0490012920 s` and `0.0023978540 s`. The trajectory is
  95.3% of their median sum, so circuit contraction, normalization, and its
  reverse-mode path take priority over a Hamiltonian-only rewrite.

- `2026-07-28T01:50:00Z`: merged upstream policy commit
  `d612cd3ae752a8d16fd0b59c717d19abd4fb5f38`, which removes hidden-controller
  and holdout requirements and explicitly permits one canonical public case
  for a fixed deterministic task. Resolved overlapping gate documentation in
  favor of the upstream public-evaluation policy while preserving the
  validated Task 05 manifest and profiler evidence. All 35 tests passed.
  `python3 research/check_gates.py --task 05 --baseline-report
  results/task-05-reference-baseline-v1/results.json --json` reported
  `research_ready: true` and `promotion_ready: true`. Candidate rounds are now
  authorized.

- Correction recorded `2026-07-28T02:05:22Z`: the authoritative timestamp for
  the preceding merge/readiness event is `2026-07-28T02:05:22Z`, not
  `2026-07-28T01:50:00Z`.

## Experiment `r01a7c9`

Task: `task-05`

Branch: `codex/orbitbreakers/task-05/r01a7c9`

Worktree:
`/Users/hmyuuu/forge/ORBIT-Q-worktrees/orbitbreakers/task-05/r01a7c9`

Campaign task: `task-05`

Live upstream PR list inspected at: `2026-07-27T18:15:35Z`

No open improvement PR targets this task: confirmed.

Every prior worktree in this campaign targets the same task: confirmed.

### Hypothesis

Putting the entire fixed 600-update Adam training process in one
`jax.lax.scan`, compiled once with `K.jit`, will remove Python-to-JAX dispatch
between updates while preserving the ten normalized TensorCircuit-NG layers,
the reverse-mode gradient, every optimizer update, and the pre-update energy
history. The reference profile attributes most runtime to compiled trajectory
work, so this is expected to be a valid diagnostic improvement rather than a
10x result.

### Parent commit and diff digest

Latest accepted parent commit:
`77d858cc082e6d8237faf756b5d3eb2493e2e9c0`

Hypothesis commit: pending pre-evaluation commit.

Candidate file: `src/solutions/task-05/solution_5.py`

Candidate SHA-256:
`c5a3a7a118ea86e42771df03ff842e9a446a8e84490a4569183bbd43ec466410`

Diff SHA-256:
`4c1d83a55c5bae57a3702a12307df987263b1f81eadcd6eb0ad4cd94ce08db14`

### Permitted data

Public dataset version: `orbitq-workloads-v20260728.1`

Public manifest SHA-256:
`f65a63b01238b569de0a1cea62af5dd0923ee1b52e9a4a7ada50c88fd8815485`

All benchmark workloads and validity rules are versioned public artifacts:
confirmed. No hidden or private evaluation data is used.

### Command, seed, and environment

Benchmark command:

`./bench run 05 --solution optimized --compare-to reference --repeat 6
--engine docker --timeout 300 --no-build --output
results/task-05-r01a7c9`

Public case selector: deterministic canonical Task 05 workload in
`datasets/public/task-05/canonical.json`.

Reference SHA-256:
`ccafe626865ee39b651adaeead86b8bf6f541e3f1426da4842da92b6a0ee015f`

Evaluator SHA-256:
`dd0742cf402827beec19328bc9cf090e80a08973cf9303fd7d524a4f4cd37402`

Docker image ID:
`sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c`

Container session ID: pending.

Pair-order pattern: odd pairs reference then candidate; even pairs candidate
then reference.

TensorCircuit-NG commit/version: pinned `tensorcircuit-py311` image specified by
the repository environment lock.

JAX/JAXLIB versions: `0.10.0` / `0.10.0`.

### Hardware and five-minute cap

Host fingerprint:
`d72d96a55e39ff10c67a820a30902dbd1b919a8f41fb4dbf95c855eac59f0013`

CPU allocation: `8`

Memory allocation: `9g`

Timeout: `300 seconds`

Measured region: evaluator-reported solution runtime.

### Result: validity, runtime, and improvement

Immutable report: pending.

Report SHA-256: pending.

```text
terminal_status: pending
valid: pending
timed_out: pending
passing_pairs: pending
reference_mean_runtime_sec: pending
reference_runtime_stderr_sec: pending
reference_median_runtime_sec: pending
candidate_mean_runtime_sec: pending
candidate_runtime_stderr_sec: pending
candidate_median_runtime_sec: pending
improvement_pct: pending
improvement_pct_stderr: pending
speedup: pending
speedup_stderr: pending
paired_speedup_ci_low: pending
paired_speedup_ci_high: pending
```

Decision: pending

### Failure signal and interpretation

Pending evaluation.

### Next pivot

If scan compilation or execution is invalid or slower, return to per-update
compilation and target the profiled trajectory contraction/normalization path.
If it is valid and faster, retain it only if the paired promotion rule passes.

### Append-only experiment corrections

Append corrections below this heading. Never rewrite an earlier result after it
has informed another experiment.
