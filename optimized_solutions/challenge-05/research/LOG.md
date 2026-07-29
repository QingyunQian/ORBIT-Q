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

- `2026-07-28T02:35:40.844503Z`: Round 1 completed with 12/12 successful
  evaluator cells and six eligible alternating matched pairs. The pre-evaluation
  hypothesis commit is
  `1eb08a96b348a836fc3725e4b59c3afe40116fd9`. The shared container/session ID
  is
  `6304e595c776e1b5c6ae9392d196438f0cbcb9959b96170493995fa1242369da`.
  The run-specific host fingerprint is
  `c7e01029d3f2a95e99c9ede05b0d6b10c0a226ea6804c5e53ac6efc308e7848e`;
  it supersedes the pre-run baseline fingerprint recorded above because the
  benchmark fingerprint includes the experiment checkout state.

- `2026-07-28T02:35:40.844503Z`: immutable report
  `results/task-05-r01a7c9/results.json`, SHA-256
  `57727d0214bc96f89e814b213c26abfd3fe24cbd2a434f3c373b359d092eb618`.
  Summary SHA-256
  `c0359c150ddabfc116ae0bbb20ab0c4fcbb43d36acc968e57bde2bbe1190f92e`;
  raw `run.log` SHA-256
  `165723d7838bd928e7a932ebc943c91e8264eb818966d2097d011b8fb15efd40`.

  ```text
  terminal_status: SUCCESS
  valid: true
  timed_out: false
  passing_pairs: 6
  reference_mean_runtime_sec: 129.23501633333333
  reference_runtime_stderr_sec: 1.636560324417641
  reference_median_runtime_sec: 130.011163
  candidate_mean_runtime_sec: 117.75056466666666
  candidate_runtime_stderr_sec: 1.071133075586679
  candidate_median_runtime_sec: 118.1662445
  improvement_pct: 8.807655093603325
  improvement_pct_stderr: 1.4946655256492916
  speedup: 1.0980332252567204
  speedup_stderr: 0.017708864694580044
  paired_speedup_ci_low: 1.052511139326026
  paired_speedup_ci_high: 1.1435553111874148
  ```

- `2026-07-28T02:35:40.844503Z`: decision `keep`. The candidate passed all
  correctness and framework-fidelity checks, lowered both mean and median
  runtime, won 6/6 pairs, and its predeclared two-sided 95% Student-t lower
  bound for paired speedup exceeds 1.0. It therefore passes the promotion rule
  at a measured `1.0980x`, but it does not support a 10x claim. The result
  confirms that loop dispatch is secondary; the next round must target the
  trajectory contraction/normalization path while retaining this scan as the
  latest accepted parent.

- `2026-07-28T02:36:00Z`: post-run `./bench verify` passed. A first
  `research/check_gates.py` invocation incorrectly supplied the paired report
  where the command expects a reference-only baseline and therefore reported
  expected row-role errors. This is a checker-invocation error, not benchmark
  evidence; the frozen reference-only baseline remains
  `results/task-05-reference-baseline-v1/results.json` in the campaign setup
  worktree and is used for the authoritative post-run gate check.

- `2026-07-28T02:36:00Z`: the authoritative post-run gate check using the
  frozen reference-only baseline passed with `research_ready: true` and
  `promotion_ready: true`.

- `2026-07-28T03:20:30.606241Z`: experiment `r02b4e1` tested the
  TensorCircuit backend `K.jaxy_scan` wrapper from accepted Round 1 commit
  `63e2712cb9f92fc15fb99667d4de5b492aea4edc`. All six pairs passed, but paired
  speedup `1.1023220198x ± 0.0451853212x` had a predeclared 95% Student-t
  interval `[0.9861694540x, 1.2184745857x]`; decision `discard`. Pinned source
  inspection confirmed the wrapper delegates to direct `jax.lax.scan`, so
  Round 1 remains accepted. Immutable report SHA-256
  `d2db4e0172416eb737b22f419fc736772dc5c57bba0693a894eedc1da851d52c`;
  full evidence archive:
  `/Users/hmyuuu/forge/OrbitBreakersCampaignArchive/task-05/r02b4e1`.

## Experiment `r03c8f2`

Task/branch: `task-05` /
`codex/orbitbreakers/task-05/r03c8f2`

Worktree:
`/Users/hmyuuu/forge/ORBIT-Q-worktrees/orbitbreakers/task-05/r03c8f2`

Live PR inspection remains `2026-07-27T18:15:35Z`; no Task 05 improvement PR
was open, and all campaign worktrees target only Task 05.

### Hypothesis

Selecting the pinned TensorCircuit-NG OMECo contractor with
`tc.set_contractor("omeco")` on top of accepted whole-training scan will find
lower-memory/lower-cost tensor contraction paths for the ten normalized
RX/RZZ layers. It preserves every gate, normalization, gradient, and Adam
update. Historical unpaired evidence reported 34.85 s versus 48.27 s on a
different host, so a repeatable improvement is plausible but a standalone 10x
gain is not.

### Frozen pre-evaluation record

Parent:
`0bfd6e22a1b5341f07d7ea2d71cd0bac9d98a24f`

Hypothesis commit: pending.

Candidate path/SHA-256:
`src/solutions/task-05/solution_5.py` /
`e5d5b8c82b4664ad0e6ac55901767501c87247a2be71f181d2c977da93b7a009`

Diff SHA-256:
`b65ce1fca0a0a16cce1ef9e001e42de3156bf0e3f7249b379095bea2f71465a9`

Public dataset/version/manifest:
`orbitq-workloads-v20260728.1` /
`f65a63b01238b569de0a1cea62af5dd0923ee1b52e9a4a7ada50c88fd8815485`

Reference SHA-256:
`ccafe626865ee39b651adaeead86b8bf6f541e3f1426da4842da92b6a0ee015f`

Evaluator SHA-256:
`dd0742cf402827beec19328bc9cf090e80a08973cf9303fd7d524a4f4cd37402`

Image:
`sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c`;
TensorCircuit-NG pinned image, OMECo `0.2.4`, JAX/JAXLIB `0.10.0`.

Command:

`./bench run 05 --solution optimized --compare-to reference --repeat 6
--engine docker --timeout 300 --no-build --output
results/task-05-r03c8f2`

Canonical public deterministic case; odd pairs reference-first, even pairs
candidate-first; 8 CPUs, 9 GiB, 300 seconds per evaluator. Session and
run-specific host fingerprint pending.

### Result and decision

Pending immutable paired evaluation. If OMECo passes the frozen promotion rule,
keep it; otherwise preserve the result and restore Round 1.

### Append-only experiment corrections

Append corrections below; never rewrite evidence used by a later round.

- `2026-07-28T03:48:02.080893Z`: Round 3 completed with 12/12 successful
  cells and six eligible matched pairs. Hypothesis commit
  `7e7d162453364592b5aa9e0ee5a3217395a9e454`; shared session
  `6ede20f4e0125ffec365dfed37077c062f494362f54ce35def74e16a7bfbf8e6`;
  host fingerprint
  `c7e01029d3f2a95e99c9ede05b0d6b10c0a226ea6804c5e53ac6efc308e7848e`.
  Immutable report `results/task-05-r03c8f2/results.json`, SHA-256
  `5e55997445cecba86b9ed05338618461aefb3b17b16471475eb8117f72fb7916`;
  summary SHA-256
  `466e7ad77f979ae229777151b546ff842374256e3df34e5376a9c40270402c6f`;
  raw log SHA-256
  `277074b780333eb4a5d18bad85d224b0ef185aab31cf578598ff42d24decb6b0`.

  ```text
  terminal_status: SUCCESS
  valid: true
  timed_out: false
  passing_pairs: 6
  reference_mean_runtime_sec: 141.1573155
  reference_runtime_stderr_sec: 3.2482550983735736
  reference_median_runtime_sec: 139.6248845
  candidate_mean_runtime_sec: 94.51216283333333
  candidate_runtime_stderr_sec: 3.951687432662029
  candidate_median_runtime_sec: 92.2143155
  improvement_pct: 32.918769645208826
  improvement_pct_stderr: 2.9668817384846737
  speedup: 1.5047045246309303
  speedup_stderr: 0.06362408292718186
  paired_speedup_ci_low: 1.3411536126879844
  paired_speedup_ci_high: 1.6682554365738762
  ```

- `2026-07-28T03:48:02.080893Z`: decision `keep`. OMECo preserved
  correctness and TensorCircuit-NG fidelity, lowered mean and median, won all
  six pairs, and its 95% lower speedup bound is above 1.0. This establishes a
  promoted `1.5047x` paired improvement over the immutable reference, not a
  10x result. The next accepted parent includes both whole-training scan and
  OMECo.

- `2026-07-28T03:48:02.080893Z`: post-run `./bench verify` and authoritative
  public gate check passed with `research_ready` and `promotion_ready` true.

- `2026-07-28T04:01:35.565855Z`: experiment `r04d2a6` tested exact
  TensorCircuit-NG `MPSCircuit` on accepted Round 3. A two-update public smoke
  test passed in `114.385935 s`, but the first canonical 600-update candidate
  cell timed out after `300 s` following a `131.699338 s` passing reference.
  Decision `timeout`; no runtime claim. Differentiable exact SVD/QR overhead is
  not viable on the pinned CPU/JAX image, so Round 3 remains accepted.
  Immutable report SHA-256
  `581907e1940bd29430d554dce6f6bb0638ed99ed6e48ad6fb49f029b45444e78`;
  full evidence archive:
  `/Users/hmyuuu/forge/OrbitBreakersCampaignArchive/task-05/r04d2a6`.

- `2026-07-28T05:04:09.226933Z`: experiment `r05e9b3` tested
  `plain-experimental` against the immutable reference. All six pairs passed,
  but candidate runtimes ranged `77.341559–243.869802 s`; paired speedup
  `1.2384425827x ± 0.1613576905x` had 95% interval
  `[0.8236594344x, 1.6532257309x]`. Decision `discard`; promoted OMECo remains
  accepted. Immutable report SHA-256
  `23dcf1b1ea32a416830fdbb6d85a7f81d75ec08ffaadc25e40f9044a3d3dc51a`;
  full evidence archive:
  `/Users/hmyuuu/forge/OrbitBreakersCampaignArchive/task-05/r05e9b3`.

## Experiment `r06f1c7`

Task/branch/worktree: `task-05` /
`codex/orbitbreakers/task-05/r06f1c7` /
`/Users/hmyuuu/forge/ORBIT-Q-worktrees/orbitbreakers/task-05/r06f1c7`.
The live PR inspection remains `2026-07-27T18:15:35Z`, with no Task 05
improvement PR; all campaign worktrees remain Task 05 only.

### Hypothesis

A deterministic `cotengra.ReusableHyperOptimizer` restricted to one greedy
trial, with a one-second bound and TensorCircuit preprocessing enabled, will
cache a reusable contraction path across the repeated circuit shapes in one
evaluator. It preserves the accepted dense TensorCircuit gates,
normalizations, gradients, and 600 updates while testing whether reusable path
lookup reduces OMECo planning/execution overhead.

### Frozen pre-evaluation record

Parent `a5266905ee29b32f2e49437966095e5d955ccaed`; hypothesis commit pending.
Candidate SHA-256
`6245d59510412fd0ffcc083f3a9653e7d245edc5ae827b56dc4fc39894691307`;
diff SHA-256
`aea1b103f9c2edac53795f3ac54daf2867e294a8ce7fb4fcd80d9ca9e49cbf9b`.
Public dataset `orbitq-workloads-v20260728.1`, manifest SHA-256
`f65a63b01238b569de0a1cea62af5dd0923ee1b52e9a4a7ada50c88fd8815485`.
Reference/evaluator SHA-256:
`ccafe626865ee39b651adaeead86b8bf6f541e3f1426da4842da92b6a0ee015f` /
`dd0742cf402827beec19328bc9cf090e80a08973cf9303fd7d524a4f4cd37402`.
Pinned image
`sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c`,
Cotengra `0.8.2`, JAX/JAXLIB `0.10.0`, 8 CPUs, 9 GiB, and 300 seconds per
evaluator.

Command: `./bench run 05 --solution optimized --compare-to reference --repeat
6 --engine docker --timeout 300 --no-build --output
results/task-05-r06f1c7`. Canonical public case; odd pairs reference-first,
even pairs candidate-first. Session and run fingerprint pending.

### Result and decision

Pending committed-candidate smoke check and immutable paired evaluation.
Promotion requires the frozen paired rule and improvement over the current
OMECo best.

### Append-only experiment corrections

Append corrections below; never rewrite prior evidence.

- `2026-07-28T05:07:55Z`: committed-candidate two-update smoke evaluation
  passed all public criteria in `3.488492 s`; compatibility evidence only.

- `2026-07-28T05:30:23.442376Z`: Round 6 completed 12/12 successful cells
  and six eligible pairs. Hypothesis commit
  `b90448438efa611855c75832d4b0e7568e3d3225`; shared session
  `ced0deee78a3125c1520bd2ab7a1c8be0d56140189fc243a01b182a698f01e31`;
  host fingerprint
  `c7e01029d3f2a95e99c9ede05b0d6b10c0a226ea6804c5e53ac6efc308e7848e`.
  Immutable report `results/task-05-r06f1c7/results.json`, SHA-256
  `77678d2fa07f5af69b9f7bc0ba14cec45ced4023d72cbc43c8bb31a4f4f9437d`;
  summary SHA-256
  `7f52cfc682db3c8710cab2f9ee897470f1fef9f565440d7d3c7f4e389acdfdf9`;
  raw log SHA-256
  `ba64ede99065a6d39e74ab4386be7731d9a08213045ee6a5889ab3d5f282d3dc`.

  ```text
  terminal_status: SUCCESS
  valid: true
  timed_out: false
  passing_pairs: 6
  reference_mean_runtime_sec: 121.443233
  reference_runtime_stderr_sec: 1.3499482750363672
  reference_median_runtime_sec: 121.020838
  candidate_mean_runtime_sec: 83.15383133333333
  candidate_runtime_stderr_sec: 0.7544856525500757
  candidate_median_runtime_sec: 83.11883900000001
  improvement_pct: 31.464737160327417
  improvement_pct_stderr: 1.2496001647278938
  speedup: 1.461524325541453
  speedup_stderr: 0.02658791609962549
  paired_speedup_ci_low: 1.393177911342711
  paired_speedup_ci_high: 1.529870739740195
  ```

- `2026-07-28T05:30:23.442376Z`: decision `keep`. The deterministic reusable
  greedy contractor passed all promotion gates and all six pairs. Its absolute
  candidate mean `83.153831 s` is `12.0%` lower than promoted OMECo Round 3's
  `94.512163 s`, with substantially lower variance, so it becomes the latest
  accepted parent. The paired result is `1.4615x`, not 10x.

- `2026-07-28T05:30:23.442376Z`: post-run benchmark verification and public
  gate check passed with `research_ready` and `promotion_ready` true.

- `2026-07-28T05:57:41.516745Z`: experiment `r07a3d8` enabled
  TensorCircuit algebraic contraction primitives on accepted Round 6. All six
  pairs passed and the 95% paired-speedup interval was
  `[1.2283702752x, 1.4251983788x]`, but candidate mean `91.053881 s`
  regressed from the accepted `83.153831 s`. Decision `discard`. Immutable
  report SHA-256
  `8a080acd9f82e5b0ff8fa26be3d8e78f553e254506fa4aa2f5c9189d5e52acf9`;
  full evidence archive:
  `/Users/hmyuuu/forge/OrbitBreakersCampaignArchive/task-05/r07a3d8`.

- `2026-07-28T06:29:36.992344Z`: experiment `r08b7e4` tested a single-array
  parameter layout. All six pairs passed and the 95% paired-speedup interval
  was `[1.2367243464x, 1.6635767533x]`, but candidate mean `91.990646 s`
  regressed from accepted Round 6's `83.153831 s`. Decision `discard`;
  immutable report SHA-256
  `0f158ffd565c6d1958e875c666409c03b97763d21a967f68a3c06995ec1d9728`;
  full evidence archive:
  `/Users/hmyuuu/forge/OrbitBreakersCampaignArchive/task-05/r08b7e4`.

- `2026-07-28T06:30:00Z`: the user requested campaign wrap-up and PR
  preparation. The campaign stops after eight isolated rounds rather than the
  originally requested twenty. Round 6 remains the campaign-best validated
  candidate. No 10x claim is made.

- `2026-07-28T06:46:44.914992Z`: the required fresh final paired rerun of
  accepted Round 6 was attempted and preserved as invalid evidence. Pair 1
  passed at `197.383894 s` reference and `108.605428 s` candidate. Pair 2's
  candidate passed at `143.108553 s`, but its immutable reference exceeded
  the `300 s` limit; the shared container stopped and no later cells ran.
  No final-rerun speedup is claimed. Report
  `results/task-05-final-r06-20260728/results.json`, SHA-256
  `89397cd1c7f791ea0ceee5f9013f1269dab99b1cd6c9377c71c1b61f6c9a4d8f`;
  summary SHA-256
  `e6b3e8cdc77cee7635580acf376b07f9f3be116e169f8a8340a9a7d3f2fa604f`;
  shared session
  `074abb0b87daa91252a5923cf34214f5ec963fe6ba4da6613315ef10f4233448`.
  The earlier complete Round 6 promotion report remains the campaign's
  eligible evidence.

- `2026-07-28T07:42:04Z`: campaign records were reorganized by task after
  wrap-up. This append-only ledger moved from root `LOG.md` to
  `research/task-05/LOG.md`; the survey, implementation comparison, profiling
  scripts, and profiler outputs moved into the same task directory. Historical
  paths above identify their locations when those events were recorded.
  Distilled reusable conclusions are now maintained separately in
  `research/task-05/INSIGHTS.md`.

## Continuation campaign: latest TensorCircuit, six-CPU host

- `2026-07-29T06:15:04Z`: started a Task 05-only continuation from accepted
  `origin/main` commit
  `5af98f27b9404c513df8eee0f4568b1512edee19` on branch
  `codex/orbitbreakers/task-05/extreme-native`. Live inspection of all six
  open `sxzgroup/ORBIT-Q` pull requests (#2 through #7) found infrastructure,
  agent-result, verifier-policy, and Challenge 07 work, but no Task 05
  improvement, optimization, performance, or runtime PR.

- `2026-07-29T06:15:04Z`: the continuation uses the existing local image
  `orbitbreakers-expert-benchmarks:tensorcircuit-py311`, image ID
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`,
  containing TensorCircuit-NG `1.8.0.dev20260726`. The same-machine protocol
  requests six CPUs and 7 GiB because the active backend is capped below the
  historical eight-CPU profile. Absolute runtime is not compared across the
  two hardware profiles.

- `2026-07-29T06:15:04Z`: pre-candidate setup rebound the already validated
  public manifest case from Task 11 to Task 05 and added per-run `--cpus` and
  `--memory` overrides to the benchmark CLI. This keeps `bench.toml`
  unchanged and makes the resource-constrained workflow reproducible. No
  task problem, evaluator, reference, candidate, environment lock, or public
  workload case changed.

- `2026-07-29T06:15:04Z`: the first isolated hypothesis is exact layer-level
  fusion of each disjoint `RX -> RX -> RZZ` sequence into one differentiable
  TensorCircuit two-qubit gate, with endpoint RX gates retained on odd layers.
  A separate higher-upside hypothesis will test an exact no-truncation MPS
  representation whose maximum bond dimension is bounded by the five
  brickwork applications per bond. These mechanisms will not be combined
  until each has independent semantic and performance evidence.

## Experiment `e02-exact-noqr-mps`

Task/branch: `task-05` /
`codex/orbitbreakers/task-05/e02-exact-noqr-mps`.

### Hypothesis

The state has a much smaller exact representation than its dense
`2^18` vector. Every RX filter is one-site and does not increase MPS rank.
Each `exp(b Z.Z)` filter has exact operator-Schmidt rank two:

`exp(b Z.Z) = cosh(b) I.I + sinh(b) Z.Z`.

A bond participates only in its parity's five layers, so every exact MPS bond
dimension is bounded by `2^5 = 32`. Applying this local bond-2 MPO contraction
without the generic `MPSCircuit.apply_MPO` QR/RQ canonicalization should avoid
the timeout observed by historical Round 4 while preserving the full state,
all ten normalizations, gradients, and 600 updates. The TFIM is evaluated as
an exact TensorCircuit backend bond-3 MPO contraction.

This experiment replaces the dense trajectory and Hamiltonian contraction as
one isolated representation mechanism. It is independent of the fused dense
gate experiment and starts from accepted parent `0cb46e3`.

### Frozen evaluation protocol

Public workload `orbitq-workloads-v20260728.3`; image ID
`sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`;
six CPUs; 7 GiB; 300-second timeout. The candidate must pass reconstructed
state, norm, initial energy, gradient, one-update, full evaluator, and static
checks before timing. Promotion requires six alternating matched pairs
against the immutable expert; the independent MPS factor contribution will
also be reported against the accepted dense parent. Frozen at
`2026-07-29T06:27:03Z`: candidate SHA-256
`f741e6cc75b8ed1e47bbedafc557d231d54db60fb064011650fc9c7da36c9ef6`;
equivalence-harness SHA-256
`c46604a554ad6107f28d724a22f3cfaa07f9cd047d4cb5d054a05bd0f17fdc91`;
pre-commit diff SHA-256
`5184e31df8c257667116db6aacc7eecab918994e6524893e08c56761eef6f877`.

Review correction: the candidate module docstring was restored byte-for-byte
from the immutable expert source. The current source SHA-256 is
`e1a0d8a13020687f0afc89867e114683c052200c955a3515c80397a6a580b24e`.
Executable statements and benchmark results are unchanged; the frozen
measurement artifact retains the SHA-256 recorded above.

### Continuation results and decisions

- `2026-07-29T06:27:37.838917Z`: latest-image immutable-reference baseline
  completed 6/6 passing runs in shared session
  `ccb7f91a95654f8ca3dacad09b1d2fc3a25286aa35ec043657a0cfc2113c046c`.
  Runtimes were `97.277727`, `97.722246`, `98.529488`, `99.783288`,
  `98.037636`, and `100.264691` seconds. Six-run mean
  `98.602512667 s`, standard error `0.483439480 s`; first-five mean
  `98.270077 s`. Raw report SHA-256
  `98f867e0ce8f6f701a271ce1ceb0f966b2374c987bde0c2e8a994d26c3a70bfe`.
  The Task 05 promotion gate passed.

- `2026-07-29T06:35:53Z`: e02 exact-MPS equivalence passed. Dense versus
  reconstructed-state maximum error `2.79397e-08`; norm error
  `4.17233e-07`; initial energy-density error `1.79052e-04`; gradient error
  `1.19507e-05`; conditioned nonzero-gradient one-Adam-update error
  `3.72529e-09`; maximum exact bond dimension 32. One analytically zero
  gradient coordinate was excluded only from the conditioned Adam parameter
  metric; it remains included in state, energy, and full-gradient checks.

- `2026-07-29T06:52:25.185319Z`: e02 final comparison completed 12/12
  passing cells and six eligible alternating pairs in shared session
  `647123a15bd9eb489fb764cd5d48bfc60150aec029dcbb4908170e7ff72f0459`.
  Immutable expert mean `97.083712 s`; exact-MPS mean `6.8984085 s`;
  paired speedup `14.075569632x ± 0.155535830x`; 95% Student-t interval
  `[13.675752052x, 14.475387211x]`; paired reduction `92.891124882%`;
  six of six candidate wins. First-five expert/candidate means
  `96.3557272/6.8841558 s` and first-five paired speedup `14.000342779x`.
  Raw report SHA-256
  `220bf9b519cc69e7996a78855308fa1e1b70b82a51d7861561cca8b5a01678c5`.
  Decision `keep` and promote e02.

- `2026-07-29T06:57:36.086471Z`: e03 absorbed RX into both rank-2 RZZ MPO
  branches and passed equivalence, but all six direct pairs regressed.
  e02 mean `7.008928 s`; fused-MPS mean `8.3105185 s`; paired speedup
  `0.843725047x`, 95% interval `[0.822186096x, 0.865263999x]`.
  Decision `discard`. Raw report SHA-256
  `645adcee4dd4f625dab6b9b8e5845f4f1f038c8f0e1a89666ef77d372783eff5`.

- `2026-07-29T07:12:35.022063Z`: e01 dense layer fusion completed 12/12
  passing cells against the accepted dense parent. Parent mean
  `68.3585177 s`; fused mean `66.9657893 s`; paired speedup
  `1.022847338x`, 95% interval `[0.983443265x, 1.062251411x]`.
  The interval crosses 1 and one pair regressed. Decision `discard`; no
  independent speedup claim. Raw report SHA-256
  `275e09b4c8e34c3094d81fd3beb7df18142171955091356603b1f914fe699b91`.

- `2026-07-29T07:13:00Z`: a final direct exact-MPS-versus-accepted-dense-parent
  six-pair run was requested, but Docker approval disconnected before the
  command started and explicitly prohibited automatic retry. No result is
  inferred from unmatched sessions. This missing factor is disclosed in the
  report; the valid overall MPS-versus-immutable comparison remains complete.

- `2026-07-29T07:16:39Z`: sanitized ablation summary and three factor figures
  generated under `research/task-05/results-20260729`. Summary SHA-256
  `0857a251ae53eff41a05760d020e962cc5dc0a290da4cdd464bbce05a67884dc`.
  Equivalence summary SHA-256
  `1f2675bd841e63658ee45b1ac3b995c6376637bb6fa363cd7e416992a344a706`.
  The continuation report was updated from the measured evidence; recompute
  its content hash after final formatting rather than using an earlier draft
  hash.
