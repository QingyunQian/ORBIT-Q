# Autoresearch entrypoint

Read `GOAL.md` in full and follow it.

Do not edit candidate or framework code until the selected-task survey and
public workload-dataset gates in `GOAL.md` pass. When one is open, work only on
its missing evidence.

After those gates pass, use one fresh Git worktree for one hypothesis, edit one
file under `src/solutions/`, run it against the immutable `reference` with
`./bench`, and maintain the worktree's tracked `LOG.md`. Before the first
worktree, inspect the live open PRs on `sxzgroup/ORBIT-Q` and select exactly one
task without an active improvement PR. Keep every worktree in this
campaign on that task. Repeated passing baselines are required before
promotion or an improvement claim, not before a candidate hypothesis.

Treat `GOAL.md` as the authority when another launcher prompt conflicts with
it.
