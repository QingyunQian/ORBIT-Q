# Autoresearch operating procedure

`GOAL.md` is the research contract. `program.md` is the Karpathy-style
entrypoint. This directory supplies the experiment ledger templates; it does
not bypass the startup gates.

## Gate the research loop and promotion separately

Before editing `src/solutions/` or a TensorCircuit-NG checkout, require:

1. `research/task-XX/SURVEY.md` is cited, covers the selected task, and says
   `Status: READY`.
2. `datasets/public/manifest.json` says `status: "ready"` and covers the
   selected task.

Promotion additionally requires the selected task to have at least six passing
matched reference/candidate pairs under one host fingerprint, image ID,
resource profile, evaluator set, and timing scope. Symmetric failures of the
initial byte-identical files are acceptable bootstrap outcomes but have no
runtime standing.

The repository starts with the knowledge/data gates closed. Do not turn
placeholders into `ready` attestations without building and validating the
underlying artifacts. Check research and promotion readiness with:

```bash
python3 research/check_gates.py \
  --task XX \
  --baseline-report results/task-XX-reference-baseline/results.json
```

The checker's `research_ready` field controls whether candidate hypotheses may
begin; `promotion_ready` additionally requires the repeated-baseline evidence.

## Select one campaign task

Inspect the live open pull requests on `sxzgroup/ORBIT-Q`. Choose exactly one
task that has no active improvement, optimization, performance, or runtime
PR. Record the selection and inspection time in
`research/task-XX/LOG.md`.

All worktrees in the campaign must target that same task. Do not switch
tasks after observing benchmark results, and do not combine tasks in a
worktree. A different task requires a separate campaign.

## Verify and establish the references

Run from the `OrbitBreakersExpertBenchmarks` repository root:

```bash
./bench verify
./bench env doctor
./bench run XX \
  --solution reference \
  --repeat 6 \
  --engine docker \
  --timeout 300 \
  --no-build \
  --output results/task-XX-reference-baseline
```

Preserve the JSON report and its SHA-256 in a campaign archive outside the
disposable experiment worktree. Do not use the contextual numbers in
`baselines/historical.json` as measured baselines. A failed reference remains
evidence; it closes only the promotion gate for that task.

## Create one worktree for one hypothesis

Run the following from the `OrbitBreakersExpertBenchmarks` repository root.
Use the campaign task, a fresh opaque ID, and the latest accepted commit:

```bash
git worktree add \
  ../OrbitBreakersExpertBenchmarks-worktrees/task-01/<opaque-id> \
  -b codex/orbitbreakers/task-01/<opaque-id> \
  <accepted-commit>
```

Enter that worktree and bootstrap its local files:

```bash
cd ../OrbitBreakersExpertBenchmarks-worktrees/task-01/<opaque-id>
mkdir -p research/task-01
cp autoresearch/LOG_TEMPLATE.md research/task-01/LOG.md
cp autoresearch/INSIGHTS_TEMPLATE.md research/task-01/INSIGHTS.md
cp autoresearch/results.template.tsv research/task-01/results.tsv
uv sync --index-url https://pypi.org/simple
./bench verify
./bench env doctor
```

Fill the hypothesis, parent commit, permitted data, environment, and five-minute
cap in `research/task-01/LOG.md` before running data-backed experiments.
Commit the task ledger and the single candidate change before evaluation:

```bash
git add research/task-01/LOG.md src/solutions/task-01/solution_1.py
git commit -m "experiment: task 01 <opaque-id>"
```

Initialize these files only once. Later worktrees start from the latest
accepted commit, inherit the tracked task ledger and insights, and append a new
experiment section. Never reuse a worktree for a second hypothesis. Never mix
two tasks in it, and keep later campaign worktrees on the same task.

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
  > research/task-01/run.log 2>&1
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

The JSON report and log files are the evidence.
`research/task-XX/results.tsv` is only an index over those files and their
hashes.

After every experiment:

1. Hash the JSON report and append one row to
   `research/task-XX/results.tsv`.
2. Append the aggregate result and interpretation to
   `research/task-XX/LOG.md`.
3. Mark `keep`, `discard`, `invalid`, `timeout`, or `crash`.
4. Update `research/task-XX/INSIGHTS.md` when the result changes a reusable
   conclusion, rejected approach, or next hypothesis.
5. Commit the sanitized task record updates as a separate evidence commit.
6. Copy the report, logs, and task-local `results.tsv` to the campaign archive.

Do not commit raw reports when they contain machine-local credentials or
unrelated secrets. Public case-level results, workload hashes, and public seeds
may be recorded.

Promote a passing experiment by review and cherry-pick. Keep a failed worktree
until its evidence and lessons have been consolidated; then remove it with
`git worktree remove`.

## Public evaluation

All campaign workloads, configurations, seeds, evaluators, and validity rules
must be public, versioned artifacts. Hidden tuning rotations, sealed holdouts,
and controller attestations are not part of this procedure.

Run a promoted candidate once more as a fresh final paired benchmark on the
immutable public workload. If tuning continues afterward, record a new
experiment and rerun the final benchmark before making a claim.

## Framework experiments

Make TensorCircuit-NG changes in a separate framework checkout and branch.
Record the base and patch commits, build inputs, dependency hashes, and image
ID. Measure reference and optimized sources under both base and patched images
so framework gains are not attributed to solution code.
