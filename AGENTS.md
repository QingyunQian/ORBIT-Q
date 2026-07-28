# OrbitBreakersExpertBenchmarks autoresearch rules

Read `GOAL.md` and `program.md` before changing solution or framework code.

## One campaign, one task

Before the first experiment, inspect the live open pull requests on
`sxzgroup/ORBIT-Q` and select one task without an active improvement PR.
Record it in `LOG.md`. Every worktree and hypothesis in the campaign must stay
on that task. Start a separate campaign instead of switching or combining
tasks.

## Gates before optimization

Do not edit `src/solutions/` until:

- `research/SURVEY.md` is complete and marked `READY`;
- `datasets/public/manifest.json` contains a validated public workload for the
  selected task and is marked `ready`.

Placeholders do not satisfy a gate.

Repeated passing reference measurements are a promotion gate, not a prerequisite
for proposing a candidate. A symmetric failure of the immutable reference and
its initial byte-identical `optimized` copy is valid bootstrap evidence. Do not
report runtime improvement for that task until reference and candidate both
pass at least six matched pairs.

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
branch per hypothesis on the campaign task. Start from the latest accepted
commit. Create tracked `LOG.md` from `autoresearch/LOG_TEMPLATE.md`; keep raw
reports, `results.tsv`, and `run.log` untracked.

Commit the hypothesis and code before evaluation. Commit sanitized evidence
separately. Preserve failures and timeouts until their report hashes and lessons
are consolidated. Promote reviewed improvements by cherry-pick.

Use paired Docker runs:

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
diagnostic. Report mean and standard error. Treat correctness and framework
fidelity as hard gates. Never claim a speedup from a failed, timed-out,
mismatched, or unpaired result.

## Public evaluation boundary

All workload configurations, seeds, evaluators, and validity rules used by a
campaign must be versioned public artifacts. Hidden tuning records, sealed
holdouts, and controller attestations are not required. Preserve a final paired
benchmark on the immutable public workload. If optimization continues after
that final run, record it as a new experiment and rerun the final benchmark.
