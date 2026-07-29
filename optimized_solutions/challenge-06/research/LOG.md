# Task 06 Autoresearch Campaign

Task: `task-06`

Campaign branch: `codex/orbitbreakers/task-06/extreme-native`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign scope

Optimize only the immutable Task 06 human-expert TensorCircuit-NG solution.
Preserve the four true continuous-time ODE blocks, all parameters, all 100 Adam
updates, and the complete output contract. Use the latest installed framework;
do not attribute host hardware differences to solution performance.

At `2026-07-29T01:26Z`, the upstream repository returned zero open pull
requests matching `Task 06`.

## Frozen provenance

- Parent: `5af98f27b9404c513df8eee0f4568b1512edee19`.
- Reference:
  `0e7fec8d11135241eb3f3501f3651f3f337e08c636407b3da8a2858c2b3d85d1`.
- Evaluator:
  `0d2dfc7f30087896fb599925f9110190a3a61358263688dbb09cc36115a23998`.
- Image:
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`.
- TensorCircuit-NG `1.8.0.dev20260726`; JAX/JAXLIB `0.10.0`; Diffrax
  `0.7.2`; Optax `0.2.8`.
- Resource profile: six CPUs, 7 GiB, network disabled, 300-second cell cap.
- Pair order: odd `reference -> candidate`, even
  `candidate -> reference`.
- Private or hidden evaluation used: no.

## Append-only campaign events

- `2026-07-29T01:27:26Z`: created the dedicated branch from current
  `origin/main`. Reference and editable source were byte-identical.
- `2026-07-29T01:28Z`: latest-image one-update compatibility run completed in
  `7.889008 s`. The ODE API is functional; the one-update evaluator
  intentionally fails convergence thresholds and is diagnostic only.
- `2026-07-29T01:29:10Z` to `2026-07-29T01:34:01Z`: completed six immutable
  canonical reference runs in one no-network container with a fresh evaluator
  process per run. All six passed. Runtimes were `43.976600`, `45.097434`,
  `44.415753`, `44.864424`, `45.481100`, and `46.387673` seconds. Mean
  `45.037164 s`, median `44.980929 s`, standard error `0.344740 s`.
  Report: `profiles/reference-baseline-six.json`;
  SHA-256
  `bdb7b985d6a0fa6bd6e7044816f9764e90b285fa9be0e6f07b79aa2587e9ff35`.
- `2026-07-29T01:35Z`: the first runner version omitted the gate checker's
  `repeat` alias and exact `docker exec` command from serialized rows. Both
  metadata fields were reconstructed deterministically from the unmodified
  checkpoint; no measurement, output, status, order, or hash changed. The
  runner now records both fields directly.
- `2026-07-29T01:36:02Z`: froze the public canonical Task 06 workload and
  source-backed survey. Candidate edits are allowed only after
  `research/check_gates.py` reports `research_ready: true`.

## Baseline interpretation

The canonical expert passes with initial energy density `-0.5182266235`, final
history energy density `-1.5754342079`, and independently evaluated sparse
ground energy density about `-1.6025561094`. Its learned analog parameters are
well inside all bounds. The small `1.9%` max/min runtime spread makes the
baseline suitable for paired screening.

## Append-only corrections

Append later corrections here; do not rewrite any result after it informs a
candidate.

## Profiling event: immutable compiled update

Recorded at `2026-07-29T01:41Z`.

Profiler: `profile_reference.py`
(`sha256:ba1a9072b8b76b439429ef7cbfb9325c4f525c531dedcc9153813d17f4f183fd`).
Report: `profiles/reference-profile.json`
(`sha256:b96624021fde8f5a5c374c1e60d1e0aa5c24c1e15d624a520f35d7c0dc243a4e`).

Lowering and compilation took `2.851043 s` and `3.194721 s`. Eight early
post-compile optimizer updates averaged `0.280501 s` and projected
`28.050100 s` for 100 executions. XLA reported about `23.52 million` FLOPs,
`101.16 MB` bytes accessed, and `18.78 MB` temporary storage per update.
The canonical evaluator mean is `45.037164 s`, so both compilation and
steady differentiated ODE execution are material; Python-only cleanup cannot
produce a large gain.

## Profiling event: Hamiltonian action

Recorded at `2026-07-29T01:42Z`.

Profiler: `profile_hamiltonian_actions.py`
(`sha256:fcc5914a4bb1feb50bd3b1ca2bbc89c3b706a17ce701e8dbafba29c1058b0e46`).
Report: `profiles/hamiltonian-action-profile.json`
(`sha256:6d864f1d3ff26ec1625edacb39347ed8ba3575d1a081d4f67aedd9c314584472`).

The TensorCircuit COO operators agreed with the expert analog action to
`8.72e-9` maximum absolute error and with the target action to `1.91e-6`.
However, native BCOO multiplication was slower on this CPU/JAX stack:

```text
analog termwise MVP: 0.249930 ms
analog sparse BCOO:  0.851897 ms
termwise/sparse:     0.293381x

target termwise MVP: 0.344365 ms
target sparse BCOO:  1.209447 ms
termwise/sparse:     0.284729x
```

Decision: `discard before candidate integration`. The installed 1.8
`PauliStringSum2MVP` implementation is already reshape/slice/broadcast based
and XLA fuses these short local Pauli sums effectively. A full sparse rewrite
has no source-independent reason to reverse a 3.4-3.5x isolated steady
regression. Do not repeat BCOO unchanged.

## Profiling event: exact digital Euler fusion

Recorded at `2026-07-29T01:44Z`.

Profiler: `profile_digital_fusion.py`
(`sha256:681c4c55ba4550db5ca2dd0729ea7c4f0495252d375102aca48cd6c2369e1d39`).
Report: `profiles/digital-fusion-profile.json`
(`sha256:48185620664abf2da8857ed17d12a148b0ed7c8f685b5ba124c2a3b3df52c113`).

Replacing each `RZ -> RY -> RZ` triple by the exactly phased TensorCircuit
`U` gate produced maximum state, energy, and gradient errors
`8.94e-8`, `9.54e-7`, and `9.65e-7`. The isolated energy-gradient steady
speedup was only `1.0042x`, but compile-plus-first-execution fell from
`2.5819 s` to `2.2421 s`. Decision: `test end to end`; this is a compile-cost
hypothesis, not a steady-execution claim.

## Frozen follow-up ODE hypotheses

Source inspection after the initial survey exposed two additional
TensorCircuit-native controls that preserve a true adaptive ODE:

1. pass `dt0=None` so Diffrax chooses its initial step instead of forcing
   `0.01` for every smooth time-independent block;
2. compare TensorCircuit's `ode_backend="jaxode"` with the current Diffrax
   path, preserving `rtol`, `atol`, and `max_steps`.

Each is isolated after the digital-fusion screen. A candidate must pass a
canonical 100-update evaluator; lower-step runs are diagnostics only.

## Experiment `e01`: exact digital Euler fusion

Candidate commit: `9484010`.

Candidate SHA-256:
`b240cf7d3db1e1ad2a820c2d375cf58514e30e53591ef081e9c5516e8f1abd90`.

Pre-edit diff SHA-256:
`cd45478134866b89de5b704476fd03d6f8f480a60f795b9a85f89cbb3b495085`.

The candidate replaces each block's 42 `RZ/RY/RZ` circuit nodes with 14
exactly phased TensorCircuit `U` nodes. It retains all three independent
angles. The exact identity audit was already frozen in
`profiles/digital-fusion-profile.json`.

```text
max_steps=10:  10.912746 s, PASS
max_steps=100: 42.412637 s, PASS
reference six-run mean: 45.037164 s
single-screen ratio: 1.06188x
```

The canonical initial energy differs from the reference baseline by
`3.58e-7`; final history energy differs by `1.17e-5`. All output shapes,
bounds, history length, energy gates, and NumPy checks pass.

Reports:

- `profiles/e01-digital-fusion-10.json`
  (`sha256:fc0b5cc821d73025443618bf771ff076ec621021ecbfe035c2ce47cb68f0d072`);
- `profiles/e01-digital-fusion-100.json`
  (`sha256:b926c0b8004e30fe62c6f65dfc80204c6cfdd849ffd11ec6066f87d1854d1f4a`).

Decision: `keep provisionally`. The canonical screen is about 5.8% below the
immutable mean and the numerical audits pass. Continue from e01 to isolate
Diffrax automatic initial-step selection.

## Experiment `e02`: Diffrax automatic initial step

Candidate commit: `a76876b`; restore commit: `670364d`.

Candidate SHA-256:
`b48f232bf75461326bec78a0bfd830f7b634c9ef7b3acbdeffaaec97bc412582`.

The only change from e01 was `dt0=None`, asking the same TensorCircuit Diffrax
path to choose its initial step automatically while preserving solver,
tolerances, and maximum steps.

```text
max_steps=10:  e01 10.912746 s, e02 10.402433 s, e02 PASS
max_steps=100: e01 42.412637 s, e02 42.361953 s, e02 PASS
canonical delta: 0.050684 s (0.12%)
```

The canonical energies and returned analog parameters were byte-for-display
identical to e01. Reports:

- `profiles/e02-dt0-auto-10.json`
  (`sha256:c0d534a50c3b0e2e36b979166309913858a004fdb552111b36d85576de079dbe`);
- `profiles/e02-dt0-auto-100.json`
  (`sha256:d9f17e8f27e26f6c85d33c03ae7377d7b961902b6b6f877c12ab094c823962d6`).

Decision: `discard`. The `0.12%` single-run canonical difference is far below
normal run noise and supplies no evidence that the automatic choice improves
the accepted e01 path. Restore the simpler expert default before the next
experiment.

## Experiment `e03`: TensorCircuit `jaxode`

Candidate commit: `a603df8`.

Candidate SHA-256:
`158063c23d609b01c9cf057d2e8574e2b0bb101e781cf770584d8aef2473e2a1`.

Pre-edit diff SHA-256:
`c78de3e12645fd9af2fa96026017a0f30cb86b04533531b614e5a5fc704fa79b`.

The candidate changes only TensorCircuit's ODE backend selector from
`diffrax` to `jaxode`. It still calls `tc.timeevol.ode_evol_global` in raw
mode with the identical vector field, two endpoint times, `rtol=atol=1e-6`,
and `max_steps=16`. It does not introduce a Trotter approximation or direct
matrix exponential.

```text
max_steps=10:  e01 10.912746 s, e03 6.866093 s, e03 PASS
max_steps=100: reference mean 45.037164 s
               e01 42.412637 s
               e03 27.747994 s, e03 PASS
canonical single-screen vs reference mean: 1.62297x
canonical single-screen vs e01:            1.52850x
```

The frozen numerical audit passed:

```text
initial energy absolute error:     7.39e-6  <= 5e-5
maximum gradient element error:    3.38e-5  <= 5e-4
post-update parameter max error:   2.98e-8
post-update energy absolute error: 4.77e-7  <= 2e-3
```

The canonical run passes all gates with initial/final history energy
`-0.5182192326 / -1.5775290728`. Reports:

- `profiles/e03-jaxode-10.json`
  (`sha256:dd9b3d4ffbf1d146f271287520d3f9103c907131815f15fcf42b944deae70450`);
- `profiles/e03-jaxode-100.json`
  (`sha256:96777324be79271c79670acd126044e53e48f7a0550834ef6552a479a8b0bdde`);
- `profiles/e03-jaxode-equivalence.json`
  (`sha256:b258da71dcd109c3bcb55a11da180a48db446b3d6a7b3e0805ef095d7a00fcf1`).

Decision: `keep`. This is the first large end-to-end gain and remains wholly
inside TensorCircuit's supported continuous-time ODE API. Continue from e03
to isolate whole-training scan.

## Final five-pair comparison and attribution

Recorded at `2026-07-29T05:13:09Z`.

Five counterbalanced pairs were run in one no-network container, using a fresh
evaluator process for every cell, six CPUs, 7 GiB memory, and the canonical
100-update workload. All ten cells passed.

```text
reference: 41.389616, 41.389259, 41.441489, 41.582510, 41.326743 s
candidate: 27.290641, 27.643293, 27.625098, 27.380896, 27.743136 s

reference mean:       41.425923 s
candidate mean:       27.536613 s
ratio of means:        1.504394x
mean paired speedup:   1.504463x
paired standard error: 0.005659x
95% paired t-interval: [1.488750x, 1.520175x]
candidate wins:        5/5
```

Report: `profiles/e03-final-five-pair.json`.

The runner returned a nonzero process status only because its legacy promotion
booleans require a six-run known-baseline gate; that policy is inapplicable to
this explicitly requested five-pair comparison. It does not indicate a
functional failure.

Attribution after reviewing all campaign evidence:

- TensorCircuit `jaxode` is the dominant factor (`1.5285x` versus e01 in the
  controlled one-change canonical screen).
- Exact Euler fusion is a smaller compile-oriented factor; the isolated steady
  improvement was only `1.0042x`.
- Diffrax `dt0=None` was neutral (`0.12%`) and was removed.
- TensorCircuit BCOO Hamiltonian actions were `3.4–3.5x` slower and were never
  integrated.

No independent percentage is assigned to whole-training scan because it is not
part of the promoted implementation. The consolidated PR-facing report is
`IMPLEMENTATION_COMPARISON.md`.
