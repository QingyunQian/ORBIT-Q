# OrbitBreakersExpertBenchmarks

This repository packages the twelve ORBIT-Q human expert TensorCircuit-NG
solutions as a reproducible runtime benchmark. It also registers the Task
01 MPO variant and Task 05 OMECo variant without replacing the publication
references.

The benchmark implements the measurement contract from
[quantum.harness issue #78](https://github.com/QuantumBFS/quantum.harness/issues/78):
run the same evaluator on the same machine and software image, reject invalid
solutions, and compare evaluator-reported runtime across repeated paired runs.

## Bootstrap and verify

Run this command from this directory:

```bash
./bench verify
./bench env doctor
```

Build the pinned image only when `env doctor` reports that it is absent:

```bash
./bench env build tensorcircuit-py311
```

The current local profile is fixed at 8 CPUs and 9 GiB because that is the
available Colima resource envelope. Start a new baseline campaign if the host,
resource limits, image ID, or dependency lock changes.

## Step 1: measure the original references

Run every immutable human answer four times and average the result:

```bash
./bench run all \
  --solution reference \
  --repeat 4 \
  --engine docker \
  --no-build \
  --output results/reference-bootstrap
```

Use at least six repeats for a balanced, reportable baseline:

```bash
./bench run all \
  --solution reference \
  --repeat 6 \
  --engine docker \
  --no-build \
  --output results/reference-baseline
```

## Step 2: compare the editable sources

`src/solutions/` starts as a byte-identical copy of `references/`. For each
task, the paired command creates one container and runs every reference and
optimized measurement inside it as a fresh Python process. Pair order alternates
`reference → optimized`, then `optimized → reference`, to balance within-pair
position. The CLI reports each runtime as `mean ± standard error` and computes:

```text
improvement_pct = 100 * (reference_mean - optimized_mean) / reference_mean
speedup         = reference_mean / optimized_mean
```

Run all tasks:

```bash
./bench run all \
  --solution optimized \
  --compare-to reference \
  --repeat 4 \
  --engine docker \
  --no-build \
  --output results/reference-vs-optimized
```

Run one task:

```bash
./bench run 05 \
  --solution optimized \
  --compare-to reference \
  --repeat 4
```

Because the initial files are identical, a small nonzero percentage is timing
noise, not an optimization claim. The CLI omits improvement and speedup when
either side fails, times out, or lacks a valid runtime.

The completed two-pair setup sweep is summarized in
`baselines/bootstrap-2026-07-27.md`. Symmetric failures are retained as
`runtime unavailable`; they do not invalidate the infrastructure smoke test.

Named expert-derived variants remain available:

```bash
./bench run 05 --solution omeco --compare-to reference --repeat 4
./bench run 01 --solution mpo --compare-to reference --repeat 4
```

An external experiment file can be measured with `--candidate`, but tracked
autoresearch work should edit the task's file under `src/solutions/`.

Use `--engine local` only when the active Python environment exactly matches
`envs/tensorcircuit-py311/requirements.lock`. Docker is the comparison default.

## Commands

| Command | Purpose |
| --- | --- |
| `./bench list` | List tasks, environments, and registered solutions |
| `./bench list --json` | Print the registry for scripts |
| `./bench verify` | Check the 12-task inventory, paths, entrypoints, and hashes |
| `./bench env doctor` | Inspect Docker, image availability, and resource fit |
| `./bench env build` | Build the pinned TensorCircuit-NG image |
| `./bench run TASK` | Run `01` through `12`, `task-XX`, or `all`; defaults to `optimized` |

`run` accepts `--repeat`, `--solution`, `--candidate`, `--compare-to`,
`--timeout`, `--engine`, `--output`, `--no-build`, and `--dry-run`. The default
repeat count comes from `bench.toml` and is currently four. The timeout is 300
seconds per evaluator process, not per task container.

## Runtime contract

Each copied evaluator imports one staged solution module and calls
`run_solution(config)`. The evaluator starts its timer immediately before that
call and stops after the solution returns its NumPy results.

The reported runtime:

- includes JAX tracing and compilation started inside `run_solution`;
- excludes module import, evaluator setup, DMRG input preparation, and exact
  reference calculations outside the timed call;
- counts only when the evaluator prints both the runtime marker and
  `Overall: PASS`.

The original evaluators return process status zero after a functional failure.
The CLI parses the pass marker instead of trusting process status.

Docker creates one long-lived container for each task. Every measurement starts
a fresh evaluator process inside that container, so the image, cgroup, mounts,
and filesystem are shared without leaking Python module or JAX state between
solutions. The reference and optimized source snapshots are staged before the
container starts. The runner records controller wall time as diagnostic data
and uses `End-to-end solution time` as the comparison metric.

For a task submission, run reference and optimized sources on the same idle
host, container session, image ID, CPU allocation, and repeat count. Use at least
six paired runs. Report each timing, mean, median, sample standard deviation,
standard error, percentage improvement, and paired speedup. A failed,
unpaired, or timed-out candidate has no eligible runtime.

All-task JSON keeps these comparison metrics per task; it does not pool runtime
ratios or paired speedups across heterogeneous tasks.

## Environment

All tasks currently select `tensorcircuit-py311`. The lock contains:

- Python 3.11;
- TensorCircuit nightly `1.7.0.dev20260618`;
- JAX and JAXLIB `0.10.0`;
- Optax `0.2.8`, Quimb `1.11.1`, Diffrax `0.7.2`;
- OMECo `0.2.4` and TensorNetwork-NG `0.5.1`.

The container runs without network access and sets `NUMBA_DISABLE_JIT=1`.
`bench.toml` currently requests 8 CPUs and 9 GiB of memory. `env doctor` rejects
a profile that exceeds the Docker backend. Each run records the image ID and
host details so mismatched comparisons can be rejected.

The installed nightly predates the official `tc.set_contractor("omeco")`
shortcut used by three human references. The tracked environment shim backports
that exact TensorCircuit-NG shortcut before evaluator import and is hashed in
every result. The copied evaluators print the unchanged timer at six-decimal
precision so standard errors are not rounded to 10 milliseconds.

## Files

```text
bench                         single CLI
bench.toml                    environment and runner defaults
tasks/task-XX/
  problem.md                  public task statement
  task.toml                   solution, evaluator, hash, and timeout registry
  evaluator/evaluate_N.py     functional evaluator; timer printed to 6 decimals
references/task-XX/      immutable publication human expert
src/solutions/task-XX/   editable optimization starting point
  variants/                   named reference-derived variants
envs/tensorcircuit-py311/     Dockerfile and dependency locks
baselines/historical.json     prior timings, for context only
baselines/bootstrap-*.md      measured setup summaries and evidence hashes
GOAL.md                       autonomous research objective
program.md                    Karpathy-style entrypoint
autoresearch/                 worktree protocol, log, and result templates
research/                     survey and frozen-measurement scaffolding
datasets/                     selected-task public workload policy
```

The parent ORBIT-Q Harbor tasks keep their solution copies because Harbor uses
each task directory as a Docker context. This directory holds the portable
expert benchmark, and `./bench verify` detects drift through SHA-256 hashes.
See `PROVENANCE.md` for source commits and paths.

## Autoresearch

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) uses a fixed
evaluator, one editable program, short measured experiments, Git checkpoints,
and an experiment ledger. `GOAL.md` maps that loop to one eligible ORBIT-Q
task per campaign and one hypothesis per fresh Git worktree.

Do not start optimization until the selected-task survey and public workload
dataset required by `GOAL.md` exist. Before starting a campaign, inspect the
open pull requests on `sxzgroup/ORBIT-Q` and choose one task that has no active
improvement PR. Bind the campaign to that one task and do not switch tasks
inside its worktrees. A symmetric failure of the reference and its initial
byte-identical candidate is acceptable bootstrap evidence; repeated passing
pairs are required later for promotion or any improvement claim. All workload
configurations, evaluators, and validity rules are public and versioned; no
hidden tuning set, sealed holdout, or trusted-controller attestation is
required.

`autoresearch/LOCAL_PRECEDENTS.md` records which conventions were adopted after
the read-only review of the local BooleanRazor and IntrQCtrl repositories, plus
which incomplete or plan-only pieces were deliberately not treated as working
infrastructure.

`research/check_gates.py` reports separate `research_ready` and
`promotion_ready` states. Research requires the cited survey, versioned public
dataset for the selected task. Promotion additionally requires a valid six-run
reference report for that task.

The optimization target is lower valid runtime. Preserve
`run_solution(config)`, TensorCircuit-NG semantics, and every functional check.
Do not edit evaluators, manifests, environment locks, workload data, or result
parsing during an experiment.
