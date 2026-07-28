# Autoresearch operating procedure

`GOAL.md` is the research contract. `program.md` is the Karpathy-style
entrypoint. This directory supplies the experiment ledger templates; it does
not bypass the startup gates.

## Gate the research loop and promotion separately

Before editing `src/solutions/` or a TensorCircuit-NG checkout, require:

1. `research/SURVEY.md` is cited, covers the campaign task, and says
   `Status: READY`.
2. `datasets/public/manifest.json` says `status: "ready"` and covers the
   campaign task.
3. A trusted controller attests that hidden tuning rotations and a sealed
   holdout exist outside every proposal worktree.

Promotion additionally requires the campaign task to have at least six
passing matched reference/candidate pairs under one host fingerprint, image
ID, resource profile, evaluator set, and timing scope. Symmetric failures of
the initial byte-identical files are acceptable bootstrap outcomes but have no
runtime standing.

The repository starts with the knowledge/data gates closed. Do not turn
placeholders into `ready` attestations without building and validating the
underlying artifacts. Check research and promotion readiness with:

```bash
python3 research/check_gates.py \
  --baseline-report results/reference-baseline/results.json \
  --controller-attestation /controller-owned/sanitized-attestation.json
```

The attestation path must be outside the checkout. The checker accepts only its
small documented aggregate schema and rejects private fields. Its
`research_ready` field controls whether candidate hypotheses may begin;
`promotion_ready` additionally requires the repeated-baseline evidence.

## Select one campaign task

Inspect the live open pull requests on `sxzgroup/ORBIT-Q`. Choose exactly one
task that has no active improvement, optimization, performance, or runtime
PR. Record the selection and inspection time in every `LOG.md`.

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

Preserve the JSON report and its SHA-256 in controller-owned run storage. Do
not use the contextual numbers in `baselines/historical.json` as measured
baselines. A failed reference remains evidence; it closes only the promotion
gate for that task.

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

Never reuse this worktree for a second hypothesis. Never mix two tasks in it,
and keep later campaign worktrees on the same task.

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

Promote a passing experiment by review and cherry-pick. Keep a failed worktree
until its evidence and lessons have been consolidated; then remove it with
`git worktree remove`.

## Hidden evaluation

The proposal process must be unable to read hidden tuning and holdout data.
An environment variable or obscure path is routing, not access control. Use a
separate controller identity, container, host, or service with no proposer
filesystem access.

The controller may return only the aggregate fields listed in `GOAL.md`. The
sealed holdout is evaluated once for a promoted candidate. Tuning after a
holdout result requires a new holdout version.

## Framework experiments

Make TensorCircuit-NG changes in a separate framework checkout and branch.
Record the base and patch commits, build inputs, dependency hashes, and image
ID. Measure reference and optimized sources under both base and patched images
so framework gains are not attributed to solution code.
