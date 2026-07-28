# Autoresearch operating procedure

`GOAL.md` is the research contract. `program.md` is the Karpathy-style
entrypoint. This directory supplies the experiment ledger templates.

## Start from any task

Every task is eligible for candidate work, regardless of incomplete research
artifacts or active upstream pull requests. In particular, none of these are
prerequisites:

- completing `research/SURVEY.md`;
- populating `datasets/public/manifest.json`;
- obtaining a controller attestation or private tuning/holdout data;
- establishing a repeated reference baseline;
- checking whether an upstream PR covers the task.

Use those artifacts when they improve a particular experiment. Their absence
does not block editing, profiling, testing, benchmarking, review, or promotion.

Runtime claims still need valid matched reference/candidate measurements under
matching provenance. Six pairs are the recommended default for claim-quality
evidence, not a condition for starting or completing candidate work. Symmetric
failures of the initial byte-identical files are useful bootstrap outcomes but
have no runtime standing.

## Select one task per hypothesis

Record one task in each experiment's `LOG.md`. Do not combine tasks in one
worktree. The next fresh hypothesis worktree may target the same task or any
other task.

## Verify and establish the references

Run from the `OrbitBreakersExpertBenchmarks` repository root:

```bash
./bench verify
./bench env doctor
./bench run all \
  --solution reference \
  --repeat 6 \
  --engine docker \
  --timeout 300 \
  --no-build \
  --output results/reference-baseline
```

Preserve the JSON report and its SHA-256 in run storage. Do not use the
contextual numbers in `baselines/historical.json` as measured baselines. A
failed reference remains diagnostic evidence but cannot support a runtime
claim.

## Create one worktree for one hypothesis

Run the following from the `OrbitBreakersExpertBenchmarks` repository root.
Use the hypothesis task, a fresh opaque ID, and the latest accepted commit:

```bash
git worktree add \
  ../OrbitBreakersExpertBenchmarks-worktrees/task-01/<opaque-id> \
  -b codex/orbitbreakers/task-01/<opaque-id> \
  <accepted-commit>
```

Enter that worktree and bootstrap its local files:

```bash
cd ../OrbitBreakersExpertBenchmarks-worktrees/task-01/<opaque-id>
cp autoresearch/LOG_TEMPLATE.md LOG.md
cp autoresearch/results.template.tsv results.tsv
uv sync --index-url https://pypi.org/simple
./bench verify
./bench env doctor
```

Fill the hypothesis, parent commit, permitted data, environment, and five-minute
cap in `LOG.md` before running data-backed experiments. Commit `LOG.md` and the
single candidate change before evaluation:

```bash
git add LOG.md src/solutions/task-01/solution_1.py
git commit -m "experiment: task 01 <opaque-id>"
```

Never reuse this worktree for a second hypothesis and never mix two tasks in
it. A later worktree may target any task.

## Run one paired experiment

The candidate is the tracked file under `src/solutions/`; the immutable
reference is under `references/`:

```bash
./bench run 01 \
  --solution optimized \
  --compare-to reference \
  --repeat 6 \
  --engine docker \
  --timeout 300 \
  --no-build \
  --output results/task-01-<opaque-id> \
  > run.log 2>&1
```

The CLI creates one container for the task and starts a fresh evaluator process
for each cell. Odd pairs run `reference → optimized`; even pairs reverse the
order. It records evaluator runtime as the primary metric and wrapper wall time
as a diagnostic. It reports runtime mean and standard error, plus paired
percentage improvement and speedup with standard errors, only when all pairs
pass.

Each cell uses a fresh process in the shared task container and has a hard
300-second limit.
Timeouts, functional failures, missing runtime markers, nonzero exits, and
crashes are experiment outcomes. Do not filter them out.

## Record immutable evidence

The JSON report and log files are the evidence. `results.tsv` is only an index
over those files and their hashes.

After every experiment:

1. Hash the JSON report and append one row to `results.tsv`.
2. Append the aggregate result and interpretation to `LOG.md`.
3. Mark `keep`, `discard`, `invalid`, `timeout`, or `crash`.
4. Commit the sanitized `LOG.md` update as a separate evidence commit.
5. Copy the report, logs, and `results.tsv` to controller-owned immutable
   storage.

Do not commit raw reports when they contain machine-local or private
information. Do not include hidden identifiers, paths, seeds, hashes, outputs,
or case-level failures in `LOG.md`.

Promote an experiment when review finds its correctness, framework fidelity,
and available evidence sufficient for the intended claim, then cherry-pick it.
Keep a failed worktree until its evidence and lessons have been consolidated;
then remove it with `git worktree remove`.

## Optional hidden evaluation

Hidden tuning and holdout infrastructure is not required. If it is used, the
proposal process must be unable to read private data. An environment variable
or obscure path is routing, not access control. Use a separate controller
identity, container, host, or service with no proposer filesystem access.

Return only sanitized aggregate fields. A team using a sealed holdout should
evaluate it once per holdout version and create a new version after tuning on a
holdout result.

## Framework experiments

Make TensorCircuit-NG changes in a separate framework checkout and branch.
Record the base and patch commits, build inputs, dependency hashes, and image
ID. Measure reference and optimized sources under both base and patched images
so framework gains are not attributed to solution code.
