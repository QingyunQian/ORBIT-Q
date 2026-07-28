# Goal: Optimize ORBIT-Q Human Expert Solutions

Use an autoresearch loop to reduce the evaluator-reported runtime of any
ORBIT-Q human expert TensorCircuit-NG solution.

Primary sources:

- [Quantum Harness Issue #78](https://github.com/QuantumBFS/quantum.harness/issues/78)
- [ORBIT-Q](https://github.com/sxzgroup/ORBIT-Q)
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch)

## Task selection

Every task in this repository is eligible for optimization at any time,
including a task covered by an open upstream pull request. Inspecting upstream
pull requests is optional context and never determines eligibility.

A research campaign may work on multiple tasks and may change targets between
hypotheses. Each hypothesis must still name one task, use its own fresh
worktree, and edit only that task's solution file so its evidence remains
auditable.

Record the task ID in the experiment's `LOG.md`. A survey, public workload
dataset, private controller, hidden tuning rotation, sealed holdout, or
repeated baseline is useful supporting infrastructure but is not a prerequisite
for editing, profiling, testing, benchmarking, or promoting a candidate.

## Acceptance target and research target

Issue #78 accepts any runtime reduction that survives repeated measurement
while preserving the task contract, functional checks, quantum semantics, and
TensorCircuit-NG framework fidelity.

The research stretch target is at least one of:

- a 10x speedup on a task;
- a 100x speedup where the task structure permits it;
- a speedup that increases with problem size under a fixed validity rule.

The stretch target does not replace the Issue #78 acceptance target. Record and
submit smaller gains when the available measurements support them.

## Research inputs

`research/SURVEY.md` and `datasets/public/manifest.json` are optional,
incremental research aids. Use the portions that are relevant to the current
hypothesis; incomplete sections, missing cases, and null versions do not block
work on any task.

When a task benefits from additional workloads, add public cases under
`datasets/public/` according to `datasets/README.md`. The bundled task
configuration and evaluator are sufficient for ordinary local optimization
experiments.

Hidden tuning and holdout infrastructure is optional. If private data is used,
the proposal process must not receive its records, identifiers, paths, seeds,
keys, credentials, per-record results, or an arbitrary query interface.
Controller-owned aggregate reports may be used when available, but no
controller attestation is required to begin or finish a local candidate.

## Worktree contract

Use one fresh Git worktree for each falsifiable hypothesis. Start from the
latest accepted commit; never recycle a prior experiment worktree. Name
branches:

```text
codex/orbitbreakers/task-XX/<opaque-id>
```

Place worktrees under:

```text
../ORBIT-Q-worktrees/orbitbreakers/task-XX/<opaque-id>
```

Do not combine two tasks or two performance hypotheses in one worktree. The
next hypothesis may target any task.

Each worktree must contain:

- an append-only, tracked `LOG.md` created from `autoresearch/LOG_TEMPLATE.md`;
- an untracked `results.tsv` created from `autoresearch/results.template.tsv`;
- an untracked `run.log`.

Fill in the hypothesis, parent commit, task, available data, and environment
before running a benchmark. Commit the hypothesis and public-safe code before
evaluation. Commit sanitized evidence separately.

Append a log entry after every baseline, candidate run, invalid result, timeout,
crash, reset, and framework rebuild. Do not delete or rewrite prior entries.
Append a correction entry when a prior entry contains an error. Keep failures
until their lessons and immutable report hashes have been consolidated.

Before removing a worktree, commit the sanitized `LOG.md`, then preserve
`LOG.md`, `results.tsv`, `run.log`, and benchmark JSON reports in appropriate
run storage.

## Allowed changes

For a solution experiment:

1. Confirm that `src/solutions/task-XX/solution_N.py` is initialized from
   `references/task-XX/solution_N.py`.
2. Edit only that task's file under `src/solutions/`.
3. Leave every file under `references/`, plus the evaluator, task manifest,
   runner, environment lock, workload data, and scoring policy unchanged.

Preserve:

```python
def run_solution(config):
    ...
```

The returned keys, shapes, numerical meaning, iteration counts, and task
semantics must satisfy the original evaluator.

Issue #78 permits a TensorCircuit-NG framework patch. Run framework work in a
separate checkout and record the exact base commit, patch commit, build inputs,
and image digest. Re-run the reference and candidate under the base and patched
images so the report separates framework gains from solution gains.

## Prohibited methods

Do not:

- edit evaluators, the `./bench` runner, hidden data, reward files, or validity rules;
- read private data through process inspection, environment dumps, error messages, timing probes, or filesystem search;
- hardcode expected outputs, hidden configurations, seeds, evaluator thresholds, or task-specific answers;
- dispatch on evaluator details to return synthetic outputs;
- replace the intended TensorCircuit-NG computation with a raw NumPy, SciPy, or JAX simulator;
- switch the central quantum computation to another quantum framework;
- skip required optimizer steps, trajectories, samples, layers, or measurements unless the task contract permits an equivalent method;
- add a dependency or framework patch without recording its exact version and rebuilding the pinned image.

General-purpose support libraries may assist TensorCircuit-NG. They may not
replace its central quantum computation.

## Runtime measurement and claims

The evaluator-reported `runtime_sec` is the optimization metric. Exploratory
runs may use any positive repeat count. For a runtime claim, compare the
unchanged reference and candidate in matched, interleaved runs on the same
host, container session, image, evaluator, workload, and declared cache state.
The recommended default is six pairs, with odd pairs running reference then
candidate and even pairs running candidate then reference.

Report the sample size and all available fields:

```text
baseline_mean_runtime_sec
baseline_runtime_stderr_sec
candidate_mean_runtime_sec
candidate_runtime_stderr_sec
baseline_median_runtime_sec
candidate_median_runtime_sec
improvement_pct
improvement_pct_stderr
paired_speedup_mean
paired_speedup_median
paired_speedup_stderr
```

Compute the mean percentage improvement as:

```text
100 * (baseline_mean_runtime_sec - candidate_mean_runtime_sec)
    / baseline_mean_runtime_sec
```

Compute each paired speedup as:

```text
baseline_runtime_sec / candidate_runtime_sec
```

A runtime comparison has standing only when the candidate and its matched
reference satisfy correctness and framework fidelity, complete without
timeout, and use matching provenance. Failed, timed-out, mismatched, or
unpaired results remain useful diagnostic evidence but do not support a
speedup claim.

Stronger claims should use enough matched pairs to characterize noise. The
recommended promotion evidence is:

1. at least six eligible matched pairs;
2. candidate mean and median runtime below the paired baseline;
3. the candidate wins at least 80 percent of matched pairs;
4. the lower bound of a predeclared confidence interval for paired speedup
   exceeds 1.0.

These recommendations do not block candidate creation, editing, profiling,
testing, benchmarking, or review.

Apply a hard 300-second limit to each evaluator process. On timeout, stop the
shared task container, mark the cell as `timeout`, and retain the incomplete
task session as failed evidence. Record actual completion time for successful
runs. Do not pad a fast run to 300 seconds.

## Experiment loop

1. Choose any task and create a fresh worktree and branch for one hypothesis.
2. Create `LOG.md` and `results.tsv` from the templates.
3. Select one hypothesis from code inspection, profiling, public sources,
   `research/SURVEY.md`, or prior sanitized evidence.
4. Record the task, hypothesis, parent commit, and data used in `LOG.md`.
5. Change one candidate concept in one `src/solutions/` file.
6. Commit the hypothesis and candidate change before running it.
7. Run public development checks and a reference/candidate comparison
   appropriate to the experiment. The recommended claim-quality command is:

   ```bash
   ./bench run XX \
     --solution optimized \
     --compare-to reference \
     --repeat 6 \
     --engine docker \
     --timeout 300 \
     --no-build \
     --output results/task-XX-<opaque-id>
   ```

8. Append the result, immutable report hash, and decision to `results.tsv` and
   `LOG.md`.
9. Commit sanitized evidence separately from the candidate commit.
10. Promote or submit a candidate when review finds its correctness,
    framework fidelity, and available runtime evidence sufficient for the
    intended claim.
11. Restore the prior best candidate after a regression or invalid result, but
    preserve the log, terminal status, and report.
12. Start the next hypothesis in a fresh worktree. It may target the same task
    or any other task.

If an implementation bug causes a crash, fix it and rerun the same hypothesis.
If the hypothesis violates semantics or cannot pass after bounded debugging,
record `crash` or `invalid`, restore the prior best candidate, and select
another hypothesis.

Do not ask whether to continue after the loop starts. Use code inspection,
public evidence, profiler output, and prior logs to select the next experiment.
Do not request hidden-case details.

## Final report

The final report should include:

- task ID and candidate commit;
- reference source hash and candidate source hash;
- framework base and patch commits, if used;
- image digest and dependency hashes;
- evaluator and workload versions when available;
- all baseline and candidate timings used for a runtime claim;
- mean, median, sample standard deviation, standard error, percentage
  improvement, and paired speedups when the sample supports them;
- functional and framework-fidelity results;
- a description of the optimization mechanism;
- ablations for a research claim;
- scaling measurements for a scaling claim.

If a private tuning set or final holdout is available, its aggregate result may
be added as independent evidence. It is not required for local optimization or
submission.

Claim 10x, 100x, or scaling advantage only when the recorded measurements
support that claim.
