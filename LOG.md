# Task 05 Autoresearch Campaign

Task: `task-05`

Campaign branch: `codex/orbitbreakers/task-05/gate-7f3c2d`

Campaign task: `task-05`

Live upstream PR list inspected at: `2026-07-27T18:15:35Z`

Open PRs observed:

- `sxzgroup/ORBIT-Q#3`: repository-wide reward aggregation policy fix.
- `sxzgroup/ORBIT-Q#2`: ForgeCode agent integration.

No open improvement, optimization, performance, or runtime PR targets Task 05.
Every hypothesis worktree in this campaign must remain on `task-05`.

## Campaign objective

Reduce evaluator-reported runtime for the immutable Task 05 human-expert
solution while preserving all functional checks, ten non-unitary cooling
layers, per-layer normalization, differentiation through normalization, and
exactly 600 Adam updates. The stretch target is a valid 10x paired speedup.

## Gate status before candidate work

Recorded at `2026-07-27T18:15:35Z` from parent commit
`46d6636881500fa8f70618b74f89353a2b6702b4`.

- Survey: closed; the repository scaffold is not Task 05 complete or `READY`.
- Public workload dataset: closed; the manifest is `not_built`.
- Trusted controller: closed; no external sanitized attestation was supplied.
- Repeated reference promotion gate: closed; no current six-run Task 05
  baseline report was supplied.

The candidate and immutable reference both have SHA-256
`ccafe626865ee39b651adaeead86b8bf6f541e3f1426da4842da92b6a0ee015f`.
No candidate code may change until all three knowledge/data/isolation gates
pass. Repeated reference evidence may be collected before that point, but no
speedup claim may be made from historical or unmatched numbers.

## Permitted data

- Task 05 problem, evaluator, immutable expert, and tracked expert-derived
  OMECo variant.
- Public repository/framework sources and documentation.
- Public workload records and aggregate benchmark reports.

No hidden tuning or holdout data, identifiers, paths, seeds, credentials,
per-case output, or controller logs may enter this worktree.

## Append-only campaign events

- `2026-07-27T18:15:35Z`: selected `task-05`; inspected the live upstream PR
  list; recorded all gates as closed; began public gate preparation only.

## Append-only corrections

Append corrections below this heading. Never rewrite an earlier result after it
has informed an experiment.

- `2026-07-27T18:18:21Z`: canonical public-workload validation completed.
  Immutable expert status `SUCCESS`, `Overall: PASS`; evaluator runtime
  `135.503222 s`; initial/final/exact energy densities
  `-1.1720402241`, `-1.3267312050`, and `-1.3268985748`. Report SHA-256
  `b5defee28534cb68cb274563a4f8c1075acc38ed2d1b6e8cb13acf401e8011b4`.
  This one run validates the public record but is not a performance baseline.

- `2026-07-28T01:16:00Z`: six-run immutable-reference promotion baseline
  completed in one pinned Docker container. All six runs passed. Runtimes:
  `101.547164`, `106.278911`, `114.090078`, `139.061275`, `122.631583`,
  `123.046707` seconds. Mean `117.775953 s`; median `118.3608305 s`;
  sample standard deviation `13.5171310916 s`; standard error
  `5.5183456601 s`; min/max `101.547164/139.061275 s`. Host fingerprint
  `d72d96a55e39ff10c67a820a30902dbd1b919a8f41fb4dbf95c855eac59f0013`;
  image ID
  `sha256:623dc47116d71b5f4e2879a61def7beada982438cb2df45de8367d92f7ec242c`.
  Report SHA-256
  `529a1839c67c55bece0b89b82ffd3583868a082192c2f357a819566ac1463b76`.
  The repeated-reference gate passes for Task 05. No improvement is claimed
  without matched candidate pairs.

- Correction recorded `2026-07-28T01:19:00Z`: the preceding baseline event's
  authoritative completion time is `2026-07-28T01:18:25.118354Z`, not
  `2026-07-28T01:16:00Z`.
