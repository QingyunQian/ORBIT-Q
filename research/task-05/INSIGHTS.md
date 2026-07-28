# Task 05 Research Insights

Task: `task-05`

Last consolidated: `2026-07-28`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Round 6 is the accepted campaign-best candidate:

- candidate: [`src/solutions/task-05/solution_5.py`](../../src/solutions/task-05/solution_5.py);
- source SHA-256:
  `6245d59510412fd0ffcc083f3a9653e7d245edc5ae827b56dc4fc39894691307`;
- mean candidate runtime: `83.153831 s`;
- mean reference runtime: `121.443233 s`;
- mean paired speedup: `1.461524x`;
- 95% Student-t interval: `1.393178x–1.529871x`;
- paired wins: `6/6`.

The result is a valid improvement on the pinned public Task 05 workload. It is
not a 10x result and is not described as a global SOTA result because no
matched external implementation was measured in the same environment.

## Preserved semantics

Every viable candidate must retain:

- the TensorCircuit-NG `|+>^18` initial state;
- ten non-unitary RX/RZZ cooling layers on the original brickwork bonds;
- normalization after every layer and gradients through each normalization;
- the same 35-term TFIM Hamiltonian MVP;
- all 20 optimized strengths and exactly 600 sequential Adam updates at
  learning rate `0.02`;
- all pre-update energies and the original `final_a`/`final_b` output contract;
- complex64 TensorCircuit/JAX semantics in the pinned image.

## Confirmed bottlenecks

- Python dispatch around 600 individually JIT-compiled optimizer steps was
  measurable overhead. Compiling the full dependent loop with `jax.lax.scan`
  improved the paired mean speedup to `1.098033x`.
- The ten-layer normalized trajectory dominates the forward computation.
  The tracked component profile attributes `95.33%` of separately measured
  forward time to that trajectory and `4.67%` to Hamiltonian evaluation.
- Tensor contraction-path selection materially affects both mean runtime and
  variance. Reusing a bounded deterministic Cotengra greedy path with
  TensorCircuit preprocessing produced the best complete paired result.
- The immutable reference profile estimates roughly `1.187 GB` of XLA memory
  traffic per gradient/update, so reducing secondary Python or parameter-tree
  overhead alone is unlikely to approach the 10x target.

Tracked profiles:

- [`profiles/reference-profile.json`](profiles/reference-profile.json)
- [`profiles/component-profile.json`](profiles/component-profile.json)

## What worked

- Whole-training `jax.lax.scan` preserved all 600 sequential updates while
  removing repeated host dispatch.
- OMECo plus the scan produced a valid `1.504705x` paired mean speedup in its
  own session, but its `94.512163 s` candidate mean was slower and more
  variable than Round 6.
- A deterministic one-trial `cotengra.ReusableHyperOptimizer` using `greedy`,
  `combo`, and TensorCircuit preprocessing reduced the accepted candidate mean
  to `83.153831 s`.

## What did not work

- `K.jaxy_scan` did not establish improvement; its paired-speedup confidence
  interval crossed `1.0`.
- Exact `MPSCircuit` timed out on the first canonical candidate cell. Its
  two-step smoke test passed, so the failure was performance rather than an
  immediate semantic mismatch.
- The `plain-experimental` contractor was unstable, with candidate runtimes
  from `77.34 s` to `243.87 s`.
- Algebraic contraction primitives passed correctness but regressed the
  candidate mean to `91.053881 s`.
- Repacking parameters into a single array passed correctness but regressed
  the candidate mean to `91.990646 s`.

Do not repeat these variants unchanged. A retry needs a new mechanism or new
profiling evidence that explains why its outcome should differ.

## Open hypotheses

The leading remaining hypothesis is layer-level kernel fusion. Each even layer
could combine its two RX gates and one RZZ gate into one two-qubit
TensorCircuit gate per disjoint bond; odd layers would additionally handle the
two endpoint RX gates. The candidate must retain every normalization and its
gradient.

Other useful follow-ups:

- profile the Round 6 compiled program rather than extrapolating only from the
  immutable reference;
- measure compilation time and steady-state contraction time separately while
  keeping evaluator runtime primary;
- test whether a precomputed deterministic contraction tree can avoid search
  without coupling the solution to evaluator internals;
- study larger public Task 05 configurations only after defining versioned
  workloads and an unchanged validity rule.

## Evidence limits

- The campaign stopped after eight rounds rather than twenty.
- The measured speedup is `1.461524x`, not 10x.
- Evidence covers the single canonical public configuration, pinned Docker
  image, and one host resource profile; it does not establish scaling.
- A fresh closeout rerun was invalid because the immutable reference timed out
  during Pair 2. It provides no speedup claim. The complete Round 6 six-pair
  session remains the eligible promotion evidence.
