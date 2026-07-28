# Task 05 Autoresearch Campaign Report

## Scope and claim

This campaign optimized only ORBIT-Q Task 05, the custom non-unitary gate
cooling workload. It compares the immutable human-expert reference with the
fastest candidate validated during eight isolated autoresearch rounds.

No matched external implementation publishes runtime for this exact workload,
evaluator, image, and CPU allocation. The optimized source is therefore called
the **campaign-best implementation**, not a global SOTA implementation.

The campaign establishes a statistically valid runtime improvement. It does
not establish the requested 10x target.

| Role | Artifact | Commit or hash |
|---|---|---|
| Immutable expert | [`references/task-05/solution_5.py`](../../references/task-05/solution_5.py) | SHA-256 `ccafe626865ee39b651adaeead86b8bf6f541e3f1426da4842da92b6a0ee015f` |
| Campaign best | [`src/solutions/task-05/solution_5.py`](../../src/solutions/task-05/solution_5.py) | SHA-256 `6245d59510412fd0ffcc083f3a9653e7d245edc5ae827b56dc4fc39894691307` |
| Candidate hypothesis | Round `r06f1c7` | Commit `b90448438efa611855c75832d4b0e7568e3d3225` |
| Promoted campaign commit | Reusable contractor | Commit `34ccc6f` |
| Promotion evidence | Six paired runs | Commit `f674179` |

## Campaign-best result

Round 6 made two execution changes:

1. It compiled all 600 Adam updates as one `jax.lax.scan`.
2. It selected a deterministic reusable Cotengra greedy contraction path with
   TensorCircuit preprocessing.

All 12 evaluator cells passed. The candidate won all six matched pairs.

| Metric | Immutable expert | Campaign best |
|---|---:|---:|
| Passing runs | 6/6 | 6/6 |
| Mean runtime | 121.443233 s | 83.153831 s |
| Median runtime | 121.020838 s | 83.118839 s |
| Runtime standard error | 1.349948 s | 0.754486 s |
| Runtime range | 116.682191–125.822734 s | 80.736110–85.237266 s |
| Mean paired speedup | — | 1.461524x |
| Paired-speedup standard error | — | 0.026588x |
| 95% Student-t interval | — | 1.393178x–1.529871x |

The ratio-of-means runtime reduction is 31.53%. The frozen paired-improvement
mean is 31.46% with standard error 1.25 percentage points.

The immutable promotion report SHA-256 is
`77678d2fa07f5af69b9f7bc0ba14cec45ced4023d72cbc43c8bb31a4f4f9437d`.
The complete evidence is recorded in [`LOG.md`](LOG.md), with distilled
cross-round lessons in [`INSIGHTS.md`](INSIGHTS.md).

### Matched timings

| Pair | Order | Reference | Candidate | Speedup |
|---:|---|---:|---:|---:|
| 1 | reference → candidate | 116.682191 s | 85.237266 s | 1.368911x |
| 2 | candidate → reference | 120.113583 s | 81.947418 s | 1.465740x |
| 3 | reference → candidate | 121.928093 s | 81.930702 s | 1.488186x |
| 4 | candidate → reference | 125.822734 s | 80.736110 s | 1.558444x |
| 5 | reference → candidate | 124.300085 s | 84.290260 s | 1.474667x |
| 6 | candidate → reference | 119.812712 s | 84.781232 s | 1.413199x |

## Implementation

### Whole-training compilation

The expert JIT-compiles one update and dispatches it from Python 600 times.
The campaign-best implementation compiles the same sequential optimizer
process as one scan:

```python
def train_loop(p, state):
    def scan_step(carry, _):
        p, state = carry
        p, state, energy = train_step(p, state)
        return (p, state), energy

    return jax.lax.scan(
        scan_step,
        (p, state),
        xs=None,
        length=config["max_steps"],
    )
```

This preserves every dependent Adam update and returns all 600 pre-update
energies without a Python list or 600 host dispatches.

### Reusable TensorCircuit contraction path

The campaign-best source configures a deterministic one-trial Cotengra greedy
optimizer:

```python
PATH_OPTIMIZER = ctg.ReusableHyperOptimizer(
    methods=["greedy"],
    minimize="combo",
    max_time=1,
    max_repeats=1,
    parallel=False,
    progbar=False,
)
tc.set_contractor(
    "custom",
    optimizer=PATH_OPTIMIZER,
    preprocessing=True,
)
```

The path is reusable for repeated circuit topologies in one evaluator process.
The bounded path search is included in `run_solution` timing. TensorCircuit
preprocessing merges compatible one-qubit work before contraction.

## Preserved scientific work

The reference and campaign-best sources both:

- construct the initial `|+>^18` state with TensorCircuit-NG;
- apply ten non-unitary RX/RZZ cooling layers on the same brickwork bonds;
- normalize after every layer and differentiate through each normalization;
- evaluate the same 35-term TensorCircuit TFIM Hamiltonian MVP;
- optimize all 20 strengths with exactly 600 Adam updates at learning rate
  `0.02`;
- return `final_a`, `final_b`, and every pre-update energy as NumPy data;
- use complex64 TensorCircuit/JAX semantics in the pinned image.

The optimization changes execution strategy without reducing the required
scientific work.

## Eight-round result table

| Round | Hypothesis | Outcome |
|---:|---|---|
| 1 | Whole-training `lax.scan` | Keep; 1.0980x, 95% CI 1.0525x–1.1436x |
| 2 | TensorCircuit `K.jaxy_scan` wrapper | Discard; lower CI 0.9862x |
| 3 | OMECo contractor plus scan | Keep; 1.5047x, candidate mean 94.5122 s |
| 4 | Exact `MPSCircuit` | Timeout on first canonical candidate cell |
| 5 | `plain-experimental` contractor | Discard; 77.34–243.87 s candidate range |
| 6 | Deterministic reusable greedy path | Keep; campaign-best 83.1538 s mean |
| 7 | Algebraic contraction primitives | Discard; 91.0539 s mean |
| 8 | Single-array parameter layout | Discard; 91.9906 s mean |

Raw logs, TSV rows, and immutable JSON reports are retained under the external
campaign archive named in `LOG.md`.

## Profiling interpretation

The immutable expert profile measured about 1.187 GB of XLA-estimated memory
traffic per gradient/update. A component split attributed 95.33% of separate
forward-component time to the ten-layer normalized trajectory and 4.67% to
Hamiltonian evaluation.

The promoted reusable contractor reduces the candidate mean below OMECo and
also removes OMECo's high variance. The evidence supports contraction-path
reuse and preprocessing as the main improvement; parameter-tree and wrapper
changes address secondary overhead.

## Final-rerun status

A fresh final paired rerun was attempted after the user requested wrap-up.
The host slowed materially: Pair 1 passed at 197.383894 s reference and
108.605428 s candidate; Pair 2's candidate passed at 143.108553 s, but its
immutable reference timed out at 300 seconds. The runner stopped the shared
container, and no later cells ran.

That rerun is invalid for a speedup claim and is retained rather than filtered
or retried. Its report SHA-256 is
`89397cd1c7f791ea0ceee5f9013f1269dab99b1cd6c9377c71c1b61f6c9a4d8f`.
The complete, eligible Round 6 promotion session above remains the campaign's
runtime evidence.

## Limits and next work

The campaign stopped after eight rounds at the user's request, rather than the
originally planned twenty. It did not reach 10x. In the eligible Round 6
session, a 10x result would require a candidate mean near 12.14 seconds; the
measured campaign-best mean is 83.15 seconds.

The highest-value remaining route is layer-level kernel fusion: each even
layer could combine two RX gates and one RZZ gate into one two-qubit
TensorCircuit gate per disjoint bond, while odd layers additionally apply RX
to the two endpoints. Any continuation must retain every normalization and its
gradient, then repeat the same six-pair promotion protocol.
