# Local autoresearch precedents

The setup was informed by read-only inspection on 2026-07-27 and rechecked on
2026-07-28:

- `/Users/hmyuuu/workspace/BooleanRazor` at
  `66738f0ff0735053889f74d4995a7ea250f1cbf7`;
- `/Users/hmyuuu/workspace/intr-qctrl` at
  `84200a22629908271efe2e39b5f04518b2be1216`;
- historical IntrQCtrl protocol commits
  `25cd7a8cd6406c840e08496a925a1bc04d865c86` and
  `8f31f9549e5c028429db7d3b4dc4e5fac33fc477`.

No file was copied from either checkout.

The 2026-07-28 recheck found both root checkouts clean on `main` at the commits
above. The mixed-case path `/Users/hmyuuu/workspace/IntrQCtrl` does not exist;
the current checkout is the lowercase `intr-qctrl` path. BooleanRazor remains
the materialized autoresearch precedent. Current IntrQCtrl is a standalone
challenge solution without `GOAL.md`, an autoresearch directory, a benchmark
runner, or a log template.

## Conventions adopted

- Freeze the survey, workload contract, measurement protocol, and baseline
  design before candidate timing.
- Use separate custodian, proposer, and evaluator roles.
- Keep hidden data genuinely unreadable by proposal processes.
- Use a fresh worktree for one falsifiable hypothesis.
- Bind one campaign to one challenge and keep every worktree on that challenge.
- Commit the hypothesis and code before evaluation, then commit sanitized
  evidence separately.
- Keep `LOG.md` append-only and retain failures, timeouts, and invalid runs.
- Bind results to source, evaluator, image, dependency, hardware, workload, and
  timeout provenance.
- Treat correctness as an eligibility gate before comparing runtime.
- Keep a flat result ledger only as an index over immutable JSON evidence.

## Deliberate ORBIT-specific choices

- `references/` and `src/solutions/` make the immutable/editable boundary
  explicit.
- `./bench` provides the single task and all-task runtime interface.
- The primary metric is the original evaluator's
  `End-to-end solution time`; controller wall time remains diagnostic.
- Comparisons reuse one container per task, start a fresh evaluator process for
  each cell, and alternate pair order.
- Issue #78 requires a valid runtime reduction. Claims of 10x, 100x, or better
  scaling remain stretch research claims with stronger evidence requirements.

## Limits of the precedents

BooleanRazor has useful protocol tests and experiment-record discipline, but
its current baseline and sealed-evaluation workflow is incomplete. IntrQCtrl's
current main branch is a standalone challenge solution, not an autoresearch
repository. Its broader blind-benchmark runner and autoresearch files were
planned historically but were not committed as working infrastructure.

This benchmark therefore implements and tests its own runner. It does not claim
that either local project supplied a turnkey implementation.
