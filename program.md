# Autoresearch entrypoint

Read `GOAL.md` in full and follow it.

Every task is immediately eligible, including a task covered by an open
upstream pull request. The survey, workload dataset, controller attestation,
hidden tuning, holdout, and repeated baselines are optional supporting
evidence, not prerequisites.

Use one fresh Git worktree for one hypothesis, edit one task file under
`src/solutions/`, run it against the immutable `reference` with `./bench`, and
maintain the worktree's tracked `LOG.md`. A later worktree may target any task.
Use matched passing comparisons for runtime claims, while allowing exploratory
runs with any stated positive repeat count.

Treat `GOAL.md` as the authority when another launcher prompt conflicts with
it.
