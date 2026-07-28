# Task 12 Autoresearch Campaign

Destination: `research/task-12/LOG.md`

Task: `task-12`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign selection and provenance

Selected task: `task-12` (variational circuit to MPS overlap optimization).

Live open pull requests on `sxzgroup/ORBIT-Q` inspected on 2026-07-28:
`#2` (ForgeCode agent solver), `#3` (scoring-policy fix), `#4` (Fable 5
agent-axis run record), `#5` (GPT-5.6 Sol benchmark results). None is an
active Task 12 solution-improvement PR, so Task 12 is eligible for this
campaign.

Precursor research disclosure: the bottleneck profiling and candidate design
for this campaign were first executed on 2026-07-28 against the
byte-equivalent ORBIT-Q publication reference in the fork
`QingyunQian/ORBIT-Q` (branch `cursor/optimize-challenge-12-f598`,
`optimized_sloutions/challenge-12/`), on the same 4 vCPU cloud VM but with
`tensorcircuit-nightly==1.8.0.dev20260726`. All figures used for claims in
this repository were re-measured here in the pinned lock environment; the
external numbers are context only and are never pooled with in-repo
measurements. See `SURVEY.md` for the porting statement.

Campaign workspace deviation: this campaign runs in a single working clone on
the campaign host with branch `cursor/task-12-batched-su4-campaign-f598`
(Cursor cloud-agent branch-naming policy) instead of the
`codex/orbitbreakers/task-12/<opaque-id>` worktree-per-hypothesis layout.
Exactly one hypothesis edits `src/solutions/task-12/solution_12.py`, so the
one-worktree-per-hypothesis isolation is preserved in substance.

Engine deviation: the campaign host has no Docker daemon, so every in-repo
measurement uses `./bench ... --engine local` with a virtual environment
installed exactly from `envs/tensorcircuit-py311/requirements.lock`
(Python 3.12.3), the environment `sitecustomize.py` on `PYTHONPATH`, and
`NUMBA_DISABLE_JIT=1`. The `GOAL.md` Gate 3 Docker protocol therefore stays
closed on this host; local-engine paired evidence is recorded below and a
Docker rerun is requested in `IMPLEMENTATION_COMPARISON.md`.

## Reference baseline (local engine)

Date: 2026-07-28

Command:

```bash
./bench run 12 --solution reference --repeat 6 --engine local \
  --timeout 300 --output results/task-12-reference-baseline-local
```

Reference SHA-256:
`10cfd516bc250633f4675653e0d8986002e56f4d5916a9c2972c1085193f5d38`

Evaluator SHA-256:
`08940a5fabfd88a957c467edabfbe6faa7b766f38b4d518557e50e94fcf3b277`

Host fingerprint:
`748423c1790b38ddbdd8eb77499b222a173b313f350e3bc35402ee8889a49dc4`
(4 vCPU Intel Xeon x86_64, 15 GiB RAM, Linux 6.12, Python 3.12.3, JAX 0.10.0,
`tensorcircuit-nightly==1.7.0.dev20260618`)

Immutable report: `results/task-12-reference-baseline-local/results.json`
(untracked; retained on the campaign host)

Report SHA-256:
`6d15f64bdf03097f1423fa16e0f434c97b2d1e44dafd5973abec0e08df004975`

Summary SHA-256:
`600753c2e09b9561afdfd6f79e9c795a7a7d626d4b906bb6149b530a0c2164fe`

```text
terminal_status: SUCCESS x 6
valid: 6/6 (Overall: PASS in every cell)
timed_out: 0
runtime_sec: 9.113186, 9.188893, 9.067477, 9.064269, 9.086606, 9.181242
mean_runtime_sec: 9.116946
median_runtime_sec: 9.099896
sample_stdev_sec: 0.055618
stderr_sec: 0.022706
min_sec: 9.064269
max_sec: 9.188893
```

Decision: `baseline`

Context only: the shared-container Docker bootstrap of 2026-07-27 measured
the same immutable reference at 11.261 ± 0.972 s under an 8-CPU/9-GiB
container on the maintainer host (`baselines/bootstrap-2026-07-27.md`), and
the ORBIT-Q publication record lists 6.12 s on an unspecified host
(`baselines/historical.json`). These numbers are not pooled with this
campaign's measurements.

## Reference bottleneck profile

Date: 2026-07-28

Scripts: `research/task-12/profile_reference.py`,
`research/task-12/profile_expm.py` (pinned lock environment, local engine
conditions). Sanitized outputs:
`profiles/reference-profile.json`, `profiles/expm-microbench.json`.

Findings recorded in `SURVEY.md`: the reference spends ~0.14 s on target
conversion, ~1.7 s on jit trace, ~3.4 s on XLA compile of an 8884-line
StableHLO module, and ~3.3 s on 5000 steps at ~0.67 ms per step. A scan step
containing only the batched 31-gate build plus gradient and Adam costs
0.381 ms with the norm-adaptive `jax.scipy.linalg.expm` versus 0.123 ms with
the candidate's fixed-order diagonal Pade(3,3); the per-gate `su4`
construction chain, not the tensor-network contraction, dominates both the
graph size and the step time.
