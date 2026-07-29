# Task 11 Autoresearch Campaign Report

## Scope and claim

This campaign optimized only ORBIT-Q Task 11, the spin-1 Haldane-chain VQE
workload (12 three-level sites, five brickwork layers, 500 Adam updates,
string-order readout). It compares the immutable human-expert reference with
one candidate (`e01`), measured with this repository's `./bench` harness in
the pinned lock environment.

No matched external implementation publishes runtime for this exact
workload, evaluator, and allocation. The optimized source is therefore
called the **campaign-best implementation**, not a global SOTA
implementation.

The campaign establishes a statistically valid local-engine runtime
improvement of 1.464x. The campaign host has no Docker daemon, so the formal
`GOAL.md` Gate 3 Docker protocol did not run here; a Docker-engine rerun of
the same six-pair protocol is the single missing step for formal promotion.
It does not reach the 10x stretch target; the workload is
memory-bandwidth-bound on the dense 3^12 state, so the achievable gain is
bounded by cutting dense-state passes rather than by compile or dispatch.

| Role | Artifact | Commit or hash |
|---|---|---|
| Immutable expert | [`references/task-11/solution_11.py`](../../../tasks/challenge-11/solution/solution_11.py) | SHA-256 `087c7a2894b4f0383bfc476f835933940cdfd2d9812f814adede3a39375b3f00` |
| Campaign best (candidate e01) | [`src/solutions/task-11/solution_11.py`](../solution_11_fused_layers_pade.py) | SHA-256 `d5faedc2859705be1b5a259f11ae78886dfeee9f5563eade841bedd88359e5dc` |
| Candidate hypothesis | Experiment `e01` | Commit `3dbe4a4c662335600fd014126d775344445d4aad` |
| Candidate evidence | Six paired runs | Report SHA-256 `65bc5909d6939deda5f0346ccfefa19cb07afd6b6a866960ac6b746a2e64440e` |
| Workload | `datasets/public/task-11/canonical.json`, dataset `orbitq-workloads-v20260728.3` | Expert validation report SHA-256 `b4ce3d311a0d4ba74ec094b8330824d07135a82f5a23c44aece9a3e2d25f7d91` |

## Campaign-best result

Experiment e01 made four execution changes and no protocol changes:

1. Exact gate fusion: compose the three per-site spin-1 rotations into one
   3x3 unitary and absorb each even-bond pair into its 9x9 entangler, so
   every layer applies 11 two-qudit unitaries instead of 47 gate
   applications.
2. All 9x9 entangler exponentials of a layer are built with one batched
   fixed 2^5 scaling-and-squaring diagonal Pade(3,3) pass.
3. The diagonal single-ion observable uses one precomputed per-basis-state
   coefficient vector instead of 12 separate `expectation` contractions.
4. All 500 Adam updates run inside one `jax.lax.scan`, and the
   post-training readout is jit-compiled.

All 12 evaluator cells passed. The candidate won all six matched pairs.

| Metric | Immutable expert | Campaign best (e01) |
|---|---:|---:|
| Passing runs | 6/6 | 6/6 |
| Mean runtime | 168.361539 s | 114.968325 s |
| Median runtime | 168.568934 s | 115.133288 s |
| Runtime standard error | 0.367342 s | 0.252797 s |
| Mean paired speedup | — | 1.464430x |
| Paired-speedup standard error | — | 0.002767x |
| 95% Student-t interval | — | 1.457317x–1.471543x |
| Ratio-of-means improvement | — | 31.713427% |

Matched timings (order alternates `reference->candidate` /
`candidate->reference`):

| Pair | Order | Reference | Candidate | Speedup |
|---:|---|---:|---:|---:|
| 1 | reference → candidate | 169.633 s | 115.438 s | 1.4695x |
| 2 | candidate → reference | 168.531 s | 115.399 s | 1.4604x |
| 3 | reference → candidate | 168.606 s | 114.940 s | 1.4669x |
| 4 | candidate → reference | 167.352 s | 115.211 s | 1.4526x |
| 5 | reference → candidate | 168.763 s | 115.056 s | 1.4668x |
| 6 | candidate → reference | 167.284 s | 113.767 s | 1.4704x |

The complete evidence is recorded in [`LOG.md`](LOG.md), with distilled
lessons in [`INSIGHTS.md`](INSIGHTS.md).

## Implementation

### Exact gate fusion

The expert applies, per layer, 36 single-site rz/ry/rz gates and 11 two-site
entanglers (47 dense-state passes). The campaign-best implementation
composes the three rotations per site into one 3x3 unitary with batched
`rz`/`ry` constructors, then absorbs each even-bond pair into its entangler
via

```python
singles = einsum("sab,sbc,scd->sad", rz2, ry, rz1)
pair = einsum("kac,kbd->kabcd", singles[0::2], singles[1::2])
even = entangler_batch(even_theta, even_phi, beta) @ pair.reshape(...)
```

Odd-bond entanglers stay separate. The layer unitary is algebraically
identical; only the dense-state application count changes (47 -> 11).

### Batched fixed-order entangler exponentials

The expert builds each 9x9 entangler with a separate `K.expm` of the
exact generator. The campaign-best implementation stacks every generator of
a layer and exponentiates the batch with a fixed 2^5 scaling-and-squaring
diagonal Pade(3,3):

```python
a = generator / 32j   # after forming -1j * generator
a2 = a @ a
odd = a @ (a2 + 60.0 * eye)
even = 12.0 * a2 + 120.0 * eye
r = jnp.linalg.solve(even - odd, even + odd)
for _ in range(5):
    r = r @ r
```

A diagonal Pade approximant of an anti-Hermitian argument is exactly
unitary. The training trajectory's maximum generator norm is 2.959
(`profiles/equivalence-check.json`), far inside the fixed-order accuracy
envelope.

### Diagonal onsite coefficient vector

The single-ion anisotropy `0.15 sum_i (S_i^z)^2` is diagonal in the
computational basis. The candidate precomputes a length-3^12 coefficient
vector once and evaluates `sum(coeffs * |amp|^2)` instead of twelve
separate `expectation` contractions, cutting forward energy from 36 ms to
17.5 ms (`profiles/reference-profile.json`).

### Whole-training scan

The expert JIT-compiles one update and dispatches it from Python 500 times.
The campaign-best implementation compiles the same sequential Adam process
as one `jax.lax.scan` and jit-compiles the post-training energy and
string-order readout. StableHLO shrinks from 8135 to 2426 lines and the
measured step from 323 ms to 268 ms.

## Preserved scientific work

The reference and candidate both:

- start from the spin-1 Neel product state `|0,2,0,2,...>` on 12 sites;
- apply the same five brickwork layers with the same per-site rz/ry/rz
  rotations and the same even/odd entangler generator
  `theta S.S + (phi - theta) SzSz + beta (S.S)^2`;
- draw the identical seeded float32 initialization
  (`default_rng(2041)`, scale 0.05);
- run exactly 500 sequential Adam updates at learning rate 0.03 and record
  the pre-update energy density for every step;
- return `energy_density_history (500,)`, `final_energy_density`, and
  `final_string_orders (3,)` as NumPy data;
- evolve the dense state and evaluate all non-diagonal observables through
  `tc.QuditCircuit` with complex64 TensorCircuit/JAX semantics.

Trajectory equivalence was audited directly
(`profiles/equivalence-check.json`): from the identical initialization the
max absolute energy-density delta over the first 100 steps is 8.0e-5 — the
complex64 noise floor after exact fusion changes float rounding order,
amplified only by optimizer dynamics.

## Profiling interpretation

The immutable expert spends ~97% of its end-to-end time inside the 500-step
loop at ~323 ms per step; jit trace (~1.3 s) and XLA compile (~2.7 s) are
small. The step is memory-bandwidth-bound on the 4.25 MB dense state: one
9x9 two-qudit contraction costs 0.5-2.9 ms depending on bond position
(transpose+gemm lowering of strided-axis einsums). The evidence supports
cutting dense-state passes (gate fusion + diagonal onsite vector) as the
main improvement; the batched Pade and scan shrink the graph and remove
dispatch, but the residual ~115 s is still dominated by unavoidable
bandwidth-bound contractions through the framework circuit path. A bare
reshape-matmul floor probe shows only ~1.7x headroom below that path and is
not shippable under the framework-fidelity rules.

## Factor ablation addendum

The original e01 benchmark changed four factors together. It establishes the
combined `1.464x` result, but cannot assign that entire gain to each bullet.
A post-merge component profiler now removes three factors independently while
keeping the candidate math and TensorCircuit execution path fixed:

| Factor | Control | Removal ablation | Measured contribution | Numerical check |
|---|---:|---:|---:|---:|
| Fixed batched Pade entanglers | 44.78 us execution | 75.91 us with batched adaptive `K.expm` | adaptive/fixed `1.695x`; lowering+compile ratio `1.416x` | max gate error `4.68e-6` |
| Diagonal onsite vector | 0.228 ms execution | 18.513 ms for 12 framework expectations | expectations/vector `81.21x` for the isolated onsite term | absolute value error `0` |
| Whole-training scan | 1.839 s for 10 candidate steps | 2.130 s for 10 dispatches of the identical compiled step | loop/scan `1.158x`; scan is 13.6% lower execution time | histories bitwise equal |

These are component timings, not numbers to multiply together. The onsite
term has the largest isolated ratio, but it is only one part of the energy
evaluation. The scan is a real secondary improvement, not the source of the
full 31.7% end-to-end reduction. The fixed Pade kernel reduces both compile
work and gate construction.

Gate fusion remains representation-coupled with the batched entanglers in the
promoted layer: even-site rotations are absorbed into the same 9x9 gates.
The directly observed structural change is 47 dense-state applications per
layer to 11, while the existing whole-state profile changes build-state time
from 80.98 ms to 73.78 ms with Pade also enabled. This report therefore does
not invent a gate-fusion-only percentage. The defensible attribution is:

1. fewer dense-state passes plus the diagonal onsite rewrite are the dominant
   bandwidth savings;
2. fixed batched Pade reduces gate-construction and compilation cost;
3. whole-training scan contributes a smaller measured dispatch saving.

Reproduction and sanitized output:

- [`profile_factor_ablation.py`](profile_factor_ablation.py)
- [`profiles/factor-ablation.json`](profiles/factor-ablation.json)

![Task 11 component-ablation plots](figures/factor-ablation.svg)

Each panel is one independently removable component. The onsite panel uses a
logarithmic axis because its isolated ratio is much larger; the ratios are not
multiplied into an end-to-end claim. Regenerate with
[`plot_factor_ablation.py`](plot_factor_ablation.py).

## Final-rerun status

The paired session recorded above is the final local-engine benchmark for
this campaign; no tuning followed it. The formal `GOAL.md` Docker
promotion protocol (one pinned container, six counterbalanced pairs,
`--engine docker`) has not run because this host has no Docker daemon. The
single remaining step for formal promotion is:

```bash
./bench env build tensorcircuit-py311
./bench run 11 --solution optimized --compare-to reference \
  --repeat 6 --engine docker --timeout 300 --no-build \
  --output results/task-11-e01-docker-promotion
```

on a Docker-capable host, expecting the same qualitative outcome.

## Limits and next work

- Local-engine evidence only; the Docker gate is deferred as above.
- One host (4 vCPU Intel Xeon cloud VM); no scaling claim; the fixed
  workload prevents scale probes by contract.
- Python 3.12.3 with the exact lock versions (the pinned image ships
  Python 3.11); a Docker rerun removes this deviation.
- The 10x stretch target would need a candidate mean near 16.8 s; the
  measured campaign-best mean is 115.0 s. Residual time is almost entirely
  bandwidth-bound dense-state contractions; a 10x path would require leaving
  the framework circuit API, which this campaign deliberately does not do.
