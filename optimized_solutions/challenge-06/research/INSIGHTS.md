# Task 06 Research Insights

Task: `task-06`

Last consolidated: `2026-07-29`

Evidence ledger: [`LOG.md`](LOG.md)

## Current best

Experiment e03 is the current best. In the final five counterbalanced
same-container pairs, the immutable expert averaged `41.425923 s` and e03
averaged `27.536613 s`. The ratio of means is `1.50439x`; mean paired speedup
is `1.50446x` with a 95% t-interval of `[1.48875x, 1.52018x]`.

## Preserved semantics

- 14-qubit Neel state and four hybrid blocks.
- Four true adaptive ODE evolutions per loss, with the original tolerances and
  maximum-step bound.
- All analog time/coupling/detuning variables and all 168 independent digital
  Euler angles.
- Seed 2026, complex64 TensorCircuit/JAX computation, target Hamiltonian,
  exactly 100 Adam updates, and every pre-update energy.
- Original NumPy output keys, shapes, and physical meanings.

## Confirmed bottlenecks

Both compilation and differentiated ODE execution are material. Immutable
profiling measured `2.851 s` lowering, `3.195 s` compilation, and
`0.2805 s` per early optimizer update, projecting `28.05 s` for 100 compiled
executions against the `45.04 s` evaluator mean.

The expected sparse-Hamiltonian shortcut is not viable on this CPU/JAX stack.
TensorCircuit BCOO multiplication was 3.4-3.5x slower than the installed
termwise MVP in isolated analog and target actions. XLA already fuses the
short reshape/slice/broadcast Pauli sums effectively.

Exact digital Euler fusion primarily affects compilation: the isolated
energy-gradient first call improved by about `0.34 s`, while steady execution
changed by only `0.4%`.

## What worked

Exact fusion of every digital `RZ -> RY -> RZ` sequence into one
phase-corrected TensorCircuit `U` gate passed state/energy/gradient audits and
reduced the canonical single screen by about 5.8%.

Switching the same TensorCircuit raw-mode continuous ODE from Diffrax to
TensorCircuit's `jaxode` backend reduced the accepted canonical screen by a
further 34.6%. Initial energy, gradient, one Adam update, the full 100-update
functional contract, tolerances, and maximum-step bound all pass.

## What did not work

The old bootstrap's raw-mode ODE API failure is obsolete in the current image
and should not be treated as a present blocker.

Do not replace the current Hamiltonian actions with TensorCircuit/JAX BCOO
unchanged. Numerical equivalence passed, but isolated steady runtime regressed
by more than 3x.

Diffrax `dt0=None` was neutral on the canonical screen (`0.12%` faster than
e01, far below noise) and produced the same displayed optimization result.
Keep the explicit framework default unless new step-count evidence explains a
reason to revisit it.

## Factor attribution

The native `jaxode` substitution is the dominant positive factor: its
one-change canonical screen was `1.5285x` faster than the accepted
digital-fusion candidate. Euler fusion is smaller and primarily lowers
compile-plus-first-execution cost. Diffrax `dt0=None` was neutral, while BCOO
Hamiltonian conversion was a clear regression. See
[`IMPLEMENTATION_COMPARISON.md`](IMPLEMENTATION_COMPARISON.md) for the
ablation table and final paired result.

## Open hypotheses

1. Whole-training `K.jaxy_scan`.
2. A direct five-pair removal test could quantify whether `jaxode` makes the
   compile-oriented digital fusion redundant.
3. Parameter-tree simplification only if later profiling supports it.
4. Diffrax solver sweep is now low priority because the native `jaxode` path
   is materially faster.

## Evidence limits

The benchmark covers one fixed public workload, one image, and one host
resource profile. It establishes neither cross-hardware performance nor global
SOTA. Factor screens other than the final promoted comparison are diagnostic
single runs or isolated microbenchmarks and are labeled accordingly.
