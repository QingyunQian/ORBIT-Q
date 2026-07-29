# Task 03 exact post-selection product contraction

## Result

The campaign-best implementation is an exact TensorCircuit-NG contraction of
the public Task 03 circuit. In six alternating same-container pairs, the
immutable expert averaged `4.101350 s` and the candidate averaged
`0.924608 s`. Mean paired speedup is **`4.435277x`**, with a two-sided 95%
Student-t interval of **`[4.287006x, 4.583549x]`**. All 12 cells passed and
the candidate won all six pairs.

| Metric | Immutable expert | Campaign best |
| --- | ---: | ---: |
| Passing runs | 6/6 | 6/6 |
| Mean runtime | 4.101350 s | 0.924608 s |
| Median runtime | 4.063782 s | 0.924754 s |
| Standard error | 0.061519 s | 0.002699 s |
| Ratio-of-means speedup | - | 4.435770x |
| Runtime reduction | - | 77.4560% |
| Mean paired speedup | - | 4.435277x |
| 95% paired-speedup CI | - | [4.287006x, 4.583549x] |

The requested first-five view is `4.105755 s` versus `0.924128 s`, with
`4.442230x` mean paired speedup. The sixth pair was predeclared and retained;
the six-pair result above is authoritative.

This is a same-host improvement over the bundled expert on the single
canonical public workload. It is a campaign/repository best, not a claim of
global or hardware-independent SOTA.

## Exact reduction

After each layer, the task projects every even-indexed qubit onto zero. Every
brickwork bond contains exactly one even qubit and one odd qubit. Therefore no
gate ever couples two qubits that survive the next projection. Inductively, the
normalized state after every measurement layer is

```text
|0>_0 tensor |psi_1> tensor |0>_2 tensor |psi_3> tensor ...
tensor |0>_10 tensor |psi_11>.
```

For each active bond, the candidate computes the exact conditional map

```text
M = <0_even| (RX_even tensor RX_odd) RZZ RXX |even_input>
```

on the corresponding two-component odd-qubit state. `even_input` is `|+>` in
the first even layer and `|0>` thereafter. The implementation constructs
`RX`, `RXX`, and `RZZ` with `tc.gates`, and contracts them with TensorCircuit
backend operations. There is no truncation, sampling, alternate framework,
hard-coded trajectory, or removed parameter.

The six event probabilities in a layer factor because the pre-measurement
state is a product over those disjoint pairs. The open TFIM expectation is
also exact:

```text
<sum Z_i Z_(i+1)> = 2 sum_(odd=1..9) <Z_odd> + <Z_11>
<sum X_i> = sum_(odd=1..11) <X_odd>.
```

All 60 conditional log probabilities, their product, all 230 independent
parameters, seed 2027, gate order, complex64 semantics, and all 300 Adam
updates remain present.

## Factor attribution

Each retained factor was timed as six alternating matched pairs against its
immediate parent. The factors are sequential; multiplying their ratios is only
descriptive because the measurements were separate container sessions.

| One-factor transition | Parent mean | Child mean | Mean paired speedup | 95% t-CI | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| expert → exact product state, same block scan | 4.010580 s | 2.416386 s | 1.659987x | [1.629466, 1.690507] | keep |
| scalar local maps → `K.vmap` local maps | 2.428061 s | 1.177346 s | 2.062555x | [2.042209, 2.082901] | keep |
| host Adam loop → `K.jaxy_scan` | 1.169997 s | 1.018775 s | 1.149293x | [1.101925, 1.196662] | keep |
| Python X/Z expansion → observable `K.vmap` | 1.008616 s | 0.923301 s | 1.092866x | [1.062560, 1.123172] | keep |
| expert → final combination | 4.101350 s | 0.924608 s | 4.435277x | [4.287006, 4.583549] | promote |

Every factor won 6/6 pairs and all 60 measured cells in this table passed.
The dominant single factor is local-map vectorization (`2.06x` over its
parent), followed by the exact product-state reduction (`1.66x`).

- [Exact product-state factor](figures/01-exact-product-state-reduction.svg)
- [Local-map vectorization](figures/02-vectorized-local-conditional-maps.svg)
- [Whole-training scan](figures/03-whole-training-tensorcircuit-scan.svg)
- [Observable vectorization](figures/04-vectorized-product-state-observables.svg)
- [Final comparison](figures/05-final-candidate-vs-immutable-expert.svg)

Machine-readable pair values and raw-report hashes are in
[`ablation-summary.json`](profiles/ablation-summary.json).

## Rejected and interaction-sensitive paths

- The first product-state prototype Python-unrolled all five cooling blocks.
  It passed but took `7.198068 s`, slower than the expert. Restoring the
  expert's block-level scan reduced the product candidate to `2.448120 s`.
  The reduction alone is not enough if its small graph is duplicated.
- Whole-training scan on the early unrolled product graph passed but took
  `8.989970 s`; after block scan and local `K.vmap` shrank the graph, the same
  factor became a confirmed `1.149293x` improvement. This is an interaction,
  not contradictory evidence.
- An exact four-term Pauli/Schmidt form for `RZZ RXX` passed the full
  equivalence audit but screened at `0.981155 s` versus `0.969300 s` for the
  direct TensorCircuit 4-by-4 gate path. It was discarded as a noise-sized
  regression, and the simpler native gate path was restored.
- Simultaneous full-state projection was not pursued after the stronger exact
  product reduction removed the full state entirely.

## Numerical audit

The final candidate was compared with the expert before performance promotion:

| Quantity | Maximum absolute error |
| --- | ---: |
| normalized 12-qubit reconstructed state | 1.565e-7 |
| sum of 60 log probabilities | 9.537e-7 |
| initial loss | 2.980e-8 |
| auxiliary observables | 2.980e-8 |
| gradient leaves | 1.490e-8 |
| Adam optimizer state | 1.513e-9 |
| post-update loss/observables | 4.768e-7 |
| complete three-step histories | 3.576e-7 |

The largest raw one-step parameter difference is `0.006323` at
`even.xx[0,1]`. Both gradients are numerical zero at complex64 precision:
`-4.83e-9` for the expert and `+4.42e-9` for the candidate. Adam turns the
rounding-dependent sign into a visible parameter difference. For entries with
gradient magnitude above `1e-7`, the maximum update difference is only
`5.88e-7`; post-update physical outputs and short histories remain below
`4.77e-7`. The full 300-step official evaluator passed in every timed cell.
This caveat is preserved in
[`final-equivalence.json`](profiles/final-equivalence.json).

One final evaluator cell reported:

```text
Initial energy density: -0.44931075
Final energy density: -1.02593505
Final success probability: 1.55883860e-02
Final mean log event probability: -6.93538189e-02
Initial loss: -0.44583511
Final loss: -1.02246737
Overall: PASS
```

## Why it is faster

Reference profiling measured `0.659 s` lowering, `1.656 s` compilation, and a
10,054-line StableHLO module for one optimizer update. Twenty compiled updates
averaged `5.57 ms`, projecting `1.670 s` for 300 host-dispatched executions.

The final whole-training graph lowers in `0.249 s`, compiles in `0.593 s`, and
contains 2,870 StableHLO lines. Its complete 300-update compiled scan executes
in `28.9 ms`. The candidate therefore improves both sides of the cold runtime:
the differentiated quantum graph is much smaller, and all optimizer dispatch
is captured by TensorCircuit's scan.

The TensorCircuit documentation recommends JAX-backed JIT/VMAP for repeated
variational workloads:
<https://tensorcircuit.readthedocs.io/en/stable/quickstart.html>.
The backend API documents the TensorCircuit wrappers used here:
<https://tensorcircuit.readthedocs.io/en/latest/api/backends/jax_backend.html>.

## Environment and reproduction

- image:
  `sha256:b059c5fa7f75702f9afbf94ec7866e102ac32afd59d25634ec0aca0fd56e2833`;
- TensorCircuit-NG `1.8.0.dev20260726`;
- JAX/JAXLIB `0.10.0`; Optax `0.2.8`; TensorNetwork `0.5.1`;
- 6 CPUs, 7 GiB, network disabled;
- one shared container per comparison, fresh evaluator per cell;
- odd pairs reference first, even pairs candidate first;
- hard 300-second cell timeout.

Run the final comparison:

```bash
./bench run 03 \
  --solution optimized \
  --compare-to reference \
  --repeat 6 \
  --engine docker \
  --cpus 6 \
  --memory 7g \
  --timeout 300 \
  --no-build \
  --output results/task-03-final-six
```

Regenerate sanitized evidence and figures:

```bash
python3 research/task-03/summarize_results.py \
  --product PATH/TO/product/results.json \
  --local-vmap PATH/TO/local-vmap/results.json \
  --training-scan PATH/TO/training-scan/results.json \
  --observable-vmap PATH/TO/observable-vmap/results.json \
  --final PATH/TO/final/results.json \
  --output-dir research/task-03
```

Run correctness and profile helpers in the fixed image:

```bash
python research/task-03/check_equivalence.py
python research/task-03/profile_reference.py
python research/task-03/profile_candidate.py
```
