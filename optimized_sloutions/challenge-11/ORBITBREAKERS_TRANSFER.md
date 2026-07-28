# OrbitBreakers Task 11 campaign transfer

Upstream target: `hmyuuu/OrbitBreakersExpertBenchmarks`  
Transport branch (full campaign history):  
`QingyunQian/ORBIT-Q` → `cursor/task-11-fused-layer-campaign-f598`  
Base: `ed382bf` (post task-12 merge on OrbitBreakers `main`)

Cloud agent token can write only to `QingyunQian/ORBIT-Q`, not to
`hmyuuu/OrbitBreakersExpertBenchmarks` or forks of it. Same flow as task-12 PR #6.

## Open the PR (from a writable OrbitBreakers fork)

```bash
git clone https://github.com/QingyunQian/OrbitBreakersExpertBenchmarks.git
cd OrbitBreakersExpertBenchmarks
git remote add upstream https://github.com/hmyuuu/OrbitBreakersExpertBenchmarks.git
git fetch upstream main
git fetch https://github.com/QingyunQian/ORBIT-Q.git cursor/task-11-fused-layer-campaign-f598
git checkout -b cursor/task-11-fused-layer-campaign-f598 FETCH_HEAD
git push -u origin cursor/task-11-fused-layer-campaign-f598
gh pr create --repo hmyuuu/OrbitBreakersExpertBenchmarks \
  --base main \
  --head QingyunQian:cursor/task-11-fused-layer-campaign-f598 \
  --title "research: task 11 fused-layer VQE campaign (1.46x local)" \
  --body-file - <<'BODY'
Task 11 autoresearch campaign: exact gate fusion + batched Pade entanglers +
diagonal onsite vector + whole-training scan.

Local-engine six-pair result (pinned lock, no Docker on host):
reference 168.36 s → candidate 114.97 s, paired speedup **1.464x ± 0.003x**
(95% Student-t CI 1.457x–1.472x), 6/6 pairs won, 12/12 PASS.

See `research/task-11/IMPLEMENTATION_COMPARISON.md` and `LOG.md`.
Docker Gate 3 promotion still needs a maintainer `--engine docker` rerun.
BODY
```

Or load the attached bundle against an OrbitBreakers clone whose `main`
contains `ed382bf`:

```bash
git fetch orbitbreakers-task-11-campaign.bundle cursor/task-11-fused-layer-campaign-f598
git checkout -b cursor/task-11-fused-layer-campaign-f598 FETCH_HEAD
```

## Headline result

| | Reference | Candidate e01 |
|---|---:|---:|
| Mean runtime | 168.362 s | 114.968 s |
| Paired speedup | — | 1.464x (CI 1.457–1.472) |
| Cells | 6/6 PASS | 6/6 PASS |
