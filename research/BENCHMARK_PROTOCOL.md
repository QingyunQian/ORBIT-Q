# ORBIT-Q Paired Runtime Benchmark Protocol

**Status: DRAFT**

This file specifies the intended Issue #78 comparison. It does not claim that
the current CLI enforces each rule. Do not use a run for a performance claim
until tests confirm the runner, matrix, and report schema implement this
protocol.

## Objective

Measure whether an optimized solution lowers the evaluator-reported
`End-to-end solution time` while preserving the task contract, functional
checks, quantum semantics, and TensorCircuit-NG framework fidelity.

The evaluator runtime is the primary metric. Controller wall time records
process overhead and timeout enforcement as diagnostic evidence.

## Pre-run freeze

Before execution, a maintainer must:

1. record the statistical method used for any formal claim;
2. populate `BASELINE_MATRIX.template.tsv` into a versioned matrix with one row
   per planned process;
3. record a digest for the exact matrix bytes;
4. freeze the reference and optimized source commits and file hashes;
5. pin the workload version, evaluator, task manifest, image, dependency lock,
   host allocation, process state, and timeout.

Changing one of these fields creates a new matrix version. A result from the
old matrix cannot fill a row in the new matrix.

## Paired interleaved execution

Use at least six pairs for each task and workload. Run the two processes in a
pair back to back, with no other benchmark process between them.

- Odd pair index: reference, then optimized.
- Even pair index: optimized, then reference.

Stage immutable snapshots of both sources before execution. Create one
long-lived Docker container for the task, then start every cell as a fresh
evaluator process inside it. Both roles therefore share the image, cgroup,
mounts, and container filesystem without sharing Python modules or JAX state.
Use the same declared cache policy for both roles and record it as
`process_state`. Do not run all reference cells before all optimized cells.

The tracked matrix fixes each `cell_id`, `pair_id`, position, order pattern,
role, and provenance before execution. The collector must retain one terminal
row for each matrix cell and reject missing, extra, or duplicate rows.

## Five-minute cap

The controller applies a hard 300-second wall limit to each evaluator process.
At the deadline it terminates the process group and records `TIMEOUT`. A
timeout cannot produce an eligible runtime.

Because killing the host-side `docker exec` client does not guarantee that its
in-container process stops, a timeout also stops and removes the shared task
container. Remaining planned cells become incomplete failed evidence; they are
not moved into a replacement container and pooled with the original session.

The runner must not chain checkpoints or retries to give one algorithm cell
more than 300 seconds. A predeclared retry for an infrastructure fault creates
a new cell and preserves the failed row.

## Timing scope

The evaluator starts its metric timer immediately before
`run_solution(config)` and stops after the function returns its result.

The evaluator runtime includes:

- tracing, compilation, cache creation, or synchronization triggered inside
  `run_solution`;
- TensorCircuit-NG execution, contraction, optimization, sampling, and result
  assembly performed inside that call.

The evaluator runtime excludes:

- image construction and dependency installation;
- candidate staging, module import, evaluator setup, workload preparation, and
  exact reference work completed before the timed call.

The controller wall cap spans the evaluator process, including excluded setup.
Reference and optimized implementations must keep equivalent work on the same
side of the metric boundary. Moving compilation or computation outside
`run_solution` invalidates the comparison.

The setup evaluator copies preserve this boundary and print the elapsed value
to six decimal places. Changing timer placement, precision, or parsing after a
baseline campaign starts creates a new protocol version.

## Validity rules

Classify a cell as `SUCCESS` only when:

- the evaluator process exits zero before the cap;
- the final runtime marker is finite and positive;
- the final `Overall` marker is `PASS`;
- the functional result and required output schema pass;
- the framework-fidelity audit passes.

Use terminal states `SUCCESS`, `TIMEOUT`, `NONZERO_EXIT`,
`FUNCTIONAL_FAILED`, `INVALID_OUTPUT`, `POLICY_FAILED`, `SETUP_FAILED`, and
`CANCELLED`. Preserve failed rows. Write `none` for an unavailable metric and
do not leave blank fields.

A pair qualifies for speedup analysis only when both cells reach `SUCCESS`.
Report the count and identity of excluded pairs. Do not discard a slow passing
pair.

During initial setup, symmetric failures from the immutable reference and its
byte-identical candidate are valid bootstrap outcomes. They demonstrate runner
parity but provide no runtime or improvement metric. A materially changed
candidate must satisfy these validity rules before its runtime is compared.

## Pinned provenance

Each matrix and result row records:

- task, workload, pair, role, run order, and solution identity;
- shared container session ID, actual container ID, and within-pair position;
- source commit, tree digest, solution hash, evaluator hash, and task-manifest
  hash;
- environment name, image digest, Dockerfile hash, and requirements-lock hash;
- environment compatibility-shim hash;
- host fingerprint, CPU and memory limits, engine, network mode, and process
  state;
- timeout, timing-scope version, timestamps, logs, artifacts, and manifest
  hashes.

Reference and optimized cells in one pair must match every provenance field
except role, solution identity, source identity, start time, and evidence
hashes. Reject a comparison when the host, image, evaluator, workload,
allocation, cache state, timeout, or timing scope differs.

## Statistics and claims

For each task, report all cell runtimes, eligible and failed pair counts, mean,
median, sample standard deviation, standard error, minimum, and maximum.
Standard error of a runtime mean is `sample_stdev / sqrt(n)` and is undefined
for one observation.

Report the ratio-of-means percentage improvement:

```text
100 * (reference_mean - optimized_mean) / reference_mean
```

Report its paired uncertainty using the method recorded for the experiment.
Also report each pairwise speedup
`reference_runtime_sec / optimized_runtime_sec` and its mean, median, sample
standard deviation, and standard error.

Report a speedup only from matched eligible pairs. Correctness and framework
fidelity are required for the comparison.

Issue #78 accepts a reproducible runtime reduction. Claims of 10x, 100x,
scaling advantage, or state-of-the-art performance require the separate
evidence and comparator rules in `GOAL.md` and `SURVEY.md`.

## Draft exit criteria

Change the status only after automated tests prove:

- one container per task, alternating pair order, and fresh-process execution;
- process-group termination at 300 seconds;
- shared-container cleanup on timeout, interruption, and setup failure;
- exact matrix-to-result cardinality;
- fail-closed validity parsing and framework-policy checks;
- complete provenance capture and canonical evidence hashes.
