# OrbitBreakersExpertBenchmarks autoresearch rules

Read `GOAL.md` and `program.md` before changing solution or framework code.

## Task eligibility

Every task is eligible for optimization, including tasks covered by open
upstream pull requests. No live-PR inspection or exclusivity check is required.

The survey, public dataset, controller attestation, hidden tuning rotations,
sealed holdout, and repeated baselines are optional supporting evidence. Their
absence or incomplete status must never block editing, profiling, testing,
benchmarking, reviewing, or promoting a candidate.

Use one task and one falsifiable hypothesis per worktree so evidence remains
auditable. A later worktree may target any task.

## Immutable benchmark surface

During a solution experiment, do not edit:

- `references/`;
- `tasks/`;
- `src/orbitbreakers_bench/`;
- `bench`, `bench.toml`, or `envs/`;
- `datasets/`;
- validity, timing, or result-parsing rules.

Edit exactly one `src/solutions/task-XX/solution_N.py`. Preserve
`run_solution(config)`, the output contract, TensorCircuit-NG semantics, and all
required work.

## Experiment discipline

Use one fresh Git worktree and `codex/orbitbreakers/task-XX/<opaque-id>`
branch per hypothesis. Start from the latest accepted commit. Create tracked
`LOG.md` from `autoresearch/LOG_TEMPLATE.md`; keep raw reports, `results.tsv`,
and `run.log` untracked.

Commit the hypothesis and code before evaluation. Commit sanitized evidence
separately. Preserve failures and timeouts until their report hashes and lessons
are consolidated. Promote reviewed improvements by cherry-pick.

For a claim-quality runtime comparison, use paired Docker runs:

```bash
./bench run XX \
  --solution optimized \
  --compare-to reference \
  --repeat 6 \
  --engine docker \
  --timeout 300 \
  --no-build
```

Use one Docker container per task, a fresh evaluator process per cell, and
alternating pair order. The evaluator runtime is primary; wrapper wall time is
diagnostic. Exploratory runs may use fewer repeats, but must state their sample
size. Report mean and standard error when the sample supports them. Correctness
and framework fidelity are required for a runtime claim. Never claim a speedup
from a failed, timed-out, mismatched, or unpaired result.

## Hidden-data boundary when private data is used

Private tuning or holdout infrastructure is optional. When it is used, proposal
agents must not gain filesystem, credential, command, log, or query access to
private records. Return only sanitized aggregate fields.
