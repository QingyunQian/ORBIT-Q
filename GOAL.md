# Goal: Optimize One ORBIT-Q Human Expert Solution

Use an autoresearch loop to reduce the evaluator-reported runtime of exactly
one selected ORBIT-Q human expert TensorCircuit-NG solution.

Primary sources:

- [Quantum Harness Issue #78](https://github.com/QuantumBFS/quantum.harness/issues/78)
- [ORBIT-Q](https://github.com/sxzgroup/ORBIT-Q)
- [Karpathy autoresearch](https://github.com/karpathy/autoresearch)

## One-task campaign scope

The survey, dataset construction, environment setup, and reference bootstrap
cover only the task selected for that autoresearch campaign.

Before creating the first campaign worktree:

1. Choose one task.
2. Record that task as `task-XX` in `LOG.md`.
3. Use the same task for every hypothesis worktree in the campaign.

Do not edit, benchmark, or promote a second task in the same campaign.
Do not switch targets after seeing results. Close the campaign, preserve its
evidence, and start a separate campaign if another task is selected.

## Acceptance target and research target

Issue #78 accepts any runtime reduction that survives repeated measurement while preserving the task contract, functional checks, quantum semantics, and TensorCircuit-NG framework fidelity.

The research stretch target is at least one of:

- a 10x speedup on a task;
- a 100x speedup where the task structure permits it;
- a speedup that increases with problem size under a fixed validity rule.

The stretch target does not replace the Issue #78 acceptance target. Record and submit smaller gains when the measurement rule confirms them.

## Roles

Use three roles with separate access:

1. The setup maintainer builds the survey, workload dataset, environment, and baseline record.
2. The proposal agent reads expert solutions and public data, edits one candidate, and receives benchmark results.
3. The trusted controller owns hidden tuning records and the sealed final holdout. It returns aggregate validity and runtime fields.

Do not grant the proposal agent controller credentials or filesystem access to private data.

## Knowledge gates before candidate code

Do not edit a candidate solution or TensorCircuit-NG source until the cited
survey, versioned workload dataset, and trusted-controller isolation boundary
are ready. If one of those gates is incomplete, stop and report the missing
items.

The repeated-reference gate below is a later promotion gate. A symmetric
failure of the byte-identical reference and initial `optimized` copy is valid
bootstrap evidence and does not block survey-driven candidate research. It
does block any runtime-improvement claim for that task until a valid repeated
reference/candidate comparison exists.

### Gate 1: cited SOTA and performance survey in `research/SURVEY.md`

Complete the scaffold at `research/SURVEY.md`. Cite papers, framework
documentation, source files, issues, and benchmark records for each claim.

Compare the current state of the art for the selected task's workloads. Record the best reported algorithmic scaling, runtime, memory use, implementation method, hardware, and software version when a source reports them. Mark missing comparisons as open evidence gaps.

Include one section for the selected task. It must identify:

- the expert algorithm and required output contract;
- the dominant operations and expected time or memory costs;
- measured or source-supported bottlenecks;
- relevant TensorCircuit-NG primitives and contractor paths;
- relevant JAX transforms, compilation behavior, vectorization, scans, sparse operations, and device placement;
- OMECo or tensor-network contraction paths where the task uses them;
- candidate optimization hypotheses and the semantic constraints that each hypothesis must preserve.

Inspect the installed framework source and APIs. Record the TensorCircuit-NG, JAX, JAXLIB, OMECo, TensorNetwork, and Quimb versions or commits used in the environment. Cite the inspected module paths and symbols.

Add a measurement section that defines the paired-run order and confidence rule used for promotion.

Do not change its status to `READY` until it covers the selected task and
contains source citations.

### Gate 2: versioned workload dataset

Build a versioned workload dataset under the policy in `datasets/README.md`.

For the selected task, the dataset must contain:

- visible public development cases;
- a hidden tuning set that the trusted controller rotates;
- a sealed final holdout that no proposal agent can query.

Cover only the selected task. Preserve its scientific semantics and
`run_solution(config)` contract. Validate every workload with that task's human
expert solution before use.

If an immutable expert cannot run in the pinned setup, preserve that terminal
result and require an independent trusted oracle to validate the workload
semantics. Do not fabricate a passing expert result, and do not use such a task
for a runtime-improvement claim until the promotion gate has valid pairs.

The public manifest at `datasets/public/manifest.json` starts with
`status: "not_built"`. Keep this gate closed until the setup maintainer adds
real cases for the selected task, computes hashes, assigns a version, validates
that task's coverage, and changes the status to `ready`.

Store hidden tuning records, holdout records, decryption keys, seeds, paths, and populated private configuration outside the Git checkout. Use `private-data.example.toml` as a shape reference. Never populate or commit that example.

### Gate 3: repeated reference baselines for promotion

Run the selected task's immutable human reference at least six times on one
host and one pinned Docker image:

```bash
./bench verify
./bench env doctor
./bench env build tensorcircuit-py311
./bench run XX \
  --solution reference \
  --repeat 6 \
  --engine docker \
  --timeout 300 \
  --no-build \
  --output results/task-XX-reference-baseline
```

Attempt the immutable reference repeatedly and preserve every terminal result.
Use one container for the selected task and a fresh evaluator process per
measurement.
Record the container session, host fingerprint, image digest, dependency
hashes, evaluator hashes, solution hashes, individual `runtime_sec` values,
mean, median, sample standard deviation, standard error, minimum, and maximum.

For an initial byte-identical smoke comparison, a matched failure or timeout is
an acceptable setup outcome and must remain visible as missing runtime. Do not
repair, filter, or invent a timing merely to open this gate.

Keep the promotion gate closed if the selected task's reference or candidate
fails, times out, uses a mismatched environment, or lacks six eligible pairs.
An actual optimized candidate must pass correctness and this promotion gate
before receiving an improvement, speedup, SOTA, or scaling claim.

## Data boundary

The proposal agent may inspect:

- the selected task's human expert solution and tracked expert-derived variants;
- the selected task's problem statement and public development records;
- `research/SURVEY.md`;
- public framework source, documentation, and APIs;
- aggregate benchmark reports that contain no private identifiers.

The proposal agent must not receive:

- hidden tuning or holdout records;
- private paths, filenames, record identifiers, seeds, hashes, keys, tokens, or credentials;
- controller command lines or stack traces that expose private data;
- per-record hidden failures, outputs, gradients, scores, or timing;
- an interface that permits arbitrary hidden-set queries.

The trusted controller may return these aggregate fields:

```text
valid
timed_out
passing_runs
mean_runtime_sec
median_runtime_sec
runtime_stderr_sec
paired_improvement_pct
paired_improvement_stderr
paired_speedup_mean
paired_speedup_median
paired_speedup_stderr
paired_speedup_ci_low
paired_speedup_ci_high
```

Redact private values from `LOG.md`, `results.tsv`, `run.log`, JSON reports, exceptions, and process output.

## Worktree contract

Use one fresh Git worktree for each falsifiable hypothesis on the campaign's
single selected task. Start from the latest accepted commit; never recycle
a prior experiment worktree. Name branches:

```text
codex/orbitbreakers/task-XX/<opaque-id>
```

Place worktrees under:

```text
../ORBIT-Q-worktrees/orbitbreakers/task-XX/<opaque-id>
```

Do not combine two tasks or two performance hypotheses in one worktree.
Every worktree in one campaign must use the same `task-XX`.

Each worktree must contain:

- an append-only, tracked `LOG.md` created from `autoresearch/LOG_TEMPLATE.md`;
- an untracked `results.tsv` created from `autoresearch/results.template.tsv`;
- an untracked `run.log`.

Fill in the hypothesis, parent commit, permitted data, and environment before
running a benchmark. Commit the hypothesis and public-safe code before
evaluation. Commit the sanitized result as a separate evidence commit.

Append a log entry after every baseline, candidate run, invalid result, timeout,
crash, reset, and framework rebuild. Do not delete or rewrite prior entries.
Append a correction entry when a prior entry contains an error. Keep failures
until their lessons and immutable report hashes have been consolidated.

Before removing a worktree, commit the sanitized `LOG.md`, then copy `LOG.md`,
`results.tsv`, `run.log`, and benchmark JSON reports to controller-owned
storage under a run-specific directory.

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

The returned keys, shapes, numerical meaning, iteration counts, and task semantics must satisfy the original evaluator.

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

General-purpose support libraries may assist TensorCircuit-NG. They may not replace its central quantum computation.

## Runtime and promotion rule

The evaluator-reported `runtime_sec` is the optimization metric. For one task,
stage both source snapshots before execution, create one Docker container, and
run every cell as a fresh evaluator process inside that container. Alternate
pair order: odd pairs run reference then candidate; even pairs run candidate
then reference. Measure matched pairs on the same host, container session,
image, evaluator, workload version, and declared cache state.

Report:

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

A candidate earns promotion when:

1. every correctness and framework-fidelity gate passes;
2. at least six matched pairs complete without timeout;
3. candidate mean and median runtime are both lower than the paired baseline;
4. the candidate wins at least 80 percent of matched pairs;
5. the lower bound of the predeclared confidence interval for paired speedup exceeds 1.0.

Use the confidence method declared in `research/SURVEY.md` before experiments
begin. Do not change it after observing candidate results.

Apply a hard 300-second limit to each evaluator process. On timeout, stop the
shared task container, mark the cell as `timeout`, and retain the incomplete
task session as failed evidence. Record actual completion time for successful
runs. Do not pad a fast run to 300 seconds.

Correctness and TensorCircuit-NG fidelity are hard gates. A fast invalid result has no runtime standing.

## Experiment loop

Run this loop after the survey, dataset, and trusted-controller knowledge gates
pass and one eligible task has been selected. A closed repeated-baseline
gate permits hypotheses but not promotion:

1. Create a fresh worktree and branch for one hypothesis on the campaign
   task.
2. Create `LOG.md` and `results.tsv` from the templates.
3. Select one hypothesis from `research/SURVEY.md` or prior sanitized evidence.
4. Record the hypothesis, parent commit, and permitted data in `LOG.md`.
5. Change one candidate concept in one `src/solutions/` file.
6. Commit the hypothesis and candidate change before running it.
7. Run the unchanged reference and candidate as at least six interleaved,
   counterbalanced pairs in one task container:

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

8. Run public development checks.
9. Submit the commit to the trusted controller for hidden tuning evaluation.
10. Append the aggregate result, immutable report hash, and decision to
    `results.tsv` and `LOG.md`.
11. Commit the sanitized evidence separately from the candidate commit.
12. Promote the experiment only when it satisfies the repeated-baseline and
    promotion rules. If the reference still has no valid runtime, record the
    candidate as `unbenchmarked` even when it restores correctness.
13. Restore the prior best candidate after a regression or invalid result, but
    preserve the log, terminal status, and report.
14. Start the next hypothesis for the same task in a new worktree.
    Continue until a human interrupts the research loop.

If an implementation bug causes a crash, fix it and rerun the same hypothesis. If the hypothesis violates semantics or cannot pass after bounded debugging, record `crash` or `invalid`, restore the prior best candidate, and select another hypothesis.

Do not ask whether to continue after the loop starts. Use survey evidence, framework source, profiler output, and prior logs to select the next experiment. Do not request hidden-case details.

## Final holdout and Issue #78 report

Submit a promoted candidate to the sealed holdout once the trusted controller approves it. Do not tune after reading a holdout result. Start a new dataset version and research run if the team chooses to continue after a holdout failure.

The final report must include:

- task ID and candidate commit;
- reference source hash and candidate source hash;
- framework base and patch commits, if used;
- image digest and dependency hashes;
- evaluator and workload versions;
- all baseline and candidate timings;
- mean, median, sample standard deviation, standard error, percentage
  improvement, and paired speedups;
- functional and framework-fidelity results;
- a description of the optimization mechanism;
- ablations for a research claim;
- scaling measurements for a scaling claim.

Claim 10x, 100x, or scaling advantage only when the recorded measurements support that claim.
