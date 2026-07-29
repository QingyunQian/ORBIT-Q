# Task 03 Autoresearch Campaign

Task: `task-03`

Campaign branch: `codex/orbitbreakers/task-03/extreme-native`

Insights: [`INSIGHTS.md`](INSIGHTS.md)

## Campaign scope and frozen provenance

Optimize only the immutable Task 03 human-expert TensorCircuit-NG solution.
At `2026-07-29T12:28Z`, the benchmark repository had no open Task 03
optimization PR.

- Parent: `d13e2591574cc1480507b00bcb33b0c6a48e6b99`.
- Reference:
  `a451480b08610e24072098171ac57364efc764ddc5e77e3d95028f009b6d5c89`.
- Evaluator:
  `b0a173f181857735b94fea3a9d03e11b595d4584b2a4bc33da80973e4b5c6d36`.
- Image:
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`.
- TensorCircuit-NG `1.8.0.dev20260726`; JAX/JAXLIB `0.10.0`; Optax `0.2.8`.
- Resource profile: 6 CPUs, 7 GiB, network disabled, 300-second cell cap.
- Pair order: odd `reference -> candidate`, even
  `candidate -> reference`.
- Private or hidden evaluation used: no.

## Append-only campaign events

- `2026-07-29T12:27Z`: created the dedicated branch from current
  `origin/main`. Reference and editable Task 03 sources were byte-identical.
- `2026-07-29T12:30Z`: reused the already-reviewed per-invocation resource
  override and factor-ablation workflow so the 6-CPU Docker backend does not
  require a tracked `bench.toml` mutation.
- `2026-07-29T12:34Z` to `2026-07-29T12:35Z`: six immutable canonical
  reference runs all passed in one no-network container. Runtimes were
  `4.262369`, `4.044382`, `4.064228`, `4.297515`, `4.053138`, and `4.200180`
  seconds; mean `4.153635 s`, median `4.132204 s`, standard error
  `0.046447 s`. Report:
  `profiles/reference-baseline-six.json`; original report SHA-256
  `da63de7d86bd158adfd720b1cd2c3ed653fac05f97ea75c505297df65eb8ed22`.
- `2026-07-29T12:38Z`: froze the public canonical Task 03 record and cited
  survey. The first candidate hypothesis is the exact product-state
  conditional-map reduction described in `SURVEY.md`.

## Baseline interpretation

The reference passes with initial/final energy density
`-0.44931078 / -1.02593327`, final success probability `1.55883934e-2`,
and final mean log event probability `-0.0693538114`. The runtime spread is
small enough for paired screening.

## Append-only corrections

Append later corrections here; do not rewrite an earlier event after it informs
another experiment.

## Profiling and experiment events

- `2026-07-29T12:41Z`: immutable profiling measured `0.658961 s` lowering,
  `1.656039 s` compilation, 10,054 StableHLO lines, and `5.5669 ms` mean
  compiled update time. Report: `profiles/reference-profile.json`.
- The initial exact product prototype passed state, gradient, and short-history
  checks, but its whole-training scan over a Python-unrolled five-block graph
  took `8.989970 s`. Decision: `discard interaction`.
- Removing whole-training scan improved that prototype to `7.198068 s` but
  remained slower than the expert. Decision: retain the exact reduction and
  restore the expert's block-level scan before judging it.
- Commit `d7a0ada` restored block `K.jaxy_scan`. A complete screen passed in
  `2.448120 s`.
- Six-pair exact-product ablation: expert `4.010580 s`, product candidate
  `2.416386 s`, mean paired speedup `1.659987x`, 95% t-CI
  `[1.629466x, 1.690507x]`, 6/6 wins, 12/12 pass. Raw report SHA-256
  `6a4dbfbbb1eaf4cba63036f26cbe9e402cd28cf7acc55381c868077a0b105fcc`.
  Decision: `keep`.
- Commit `5defa73` vectorized the six/five independent local conditional maps.
  Six-pair parent/candidate means `2.428061 / 1.177346 s`; mean paired
  speedup `2.062555x`, 95% CI `[2.042209x, 2.082901x]`, 6/6 wins.
  Raw report SHA-256
  `874c67138cb7b8b2fa98cadc716f6cc3e93e77720448597900c5fa57d4df5560`.
  Decision: `keep`.
- Commit `95c7412` retested whole-training `K.jaxy_scan` on the reduced graph.
  Six-pair parent/candidate means `1.169997 / 1.018775 s`; paired speedup
  `1.149293x`, 95% CI `[1.101925x, 1.196662x]`, 6/6 wins. Raw report
  SHA-256
  `2e1cf1079174db8e9dc5dbd32e799560e4f6789615b979de92a5c85478edfa47`.
  Decision: `keep`.
- Commit `7f1d2f8` vectorized the six product-state X/Z expectations. Six-pair
  means `1.008616 / 0.923301 s`; paired speedup `1.092866x`, 95% CI
  `[1.062560x, 1.123172x]`, 6/6 wins. Raw report SHA-256
  `47818d35080113300f25273c1518a5a82bdb32489b92de75c7d05ffe65843f27`.
  Decision: `keep`.
- Commit `553bcb8` replaced native 4-by-4 gates with an exact four-term
  Pauli/Schmidt conditional contraction. Equivalence passed, but screens were
  `0.981155 s` versus `0.969300 s` for the direct gate path. Commit `883c9fd`
  restored the simpler native gate implementation. Decision: `discard`.
- Final six-pair report: expert `4.101350 s`, candidate `0.924608 s`;
  paired speedup `4.435277x`, 95% CI `[4.287006x, 4.583549x]`, 6/6 wins,
  12/12 pass. First-five means are `4.105755 / 0.924128 s`. Raw report
  SHA-256
  `0ab04755552fe85411d2f81155517e6835962e5b460264bc8496e7cb30f9873d`.
  Decision: `promote`.
- Final equivalence: state `1.565e-7`, log-probability sum `9.537e-7`,
  gradient `1.490e-8`, material-gradient Adam update `5.877e-7`,
  post-update physical values `4.768e-7`, and short histories `3.576e-7`.
  The raw `0.006323` parameter discrepancy is confined to opposite signs of
  order-`1e-9` numerical-zero gradients and does not affect the output
  trajectory at the frozen tolerance.
- Final profiling measured `0.249376 s` lowering, `0.593235 s` compilation,
  2,870 StableHLO lines, and `0.028853 s` execution for all 300 updates.
