# Challenge 12: 4.2x Same-Protocol Speedup of the Reference Solution

This directory contains two performance-optimized variants of the published
reference `tasks/challenge-12/solution/solution_12.py` (variational circuit to
MPS overlap optimization, 32 qubits, 5000 Adam updates). The physics protocol
is unchanged in both: identical su(4)-generator parameterization of every SU4
gate, identical seed and initialization, identical Adam schedule, identical
direct tensor-network overlap loss against the evaluator-provided DMRG-MPS
target. Only the computation is restructured.

- `solution_12_batched.py` -- minimal-diff variant: keeps the reference's
  32-qubit `tc.Circuit` construction line by line and changes only how the
  gate matrices are built and how the loop runs (optimizations 1 + 3 below).
- `solution_12_fused.py` -- fastest variant: additionally contracts the
  brickwork on pair-fused four-level sites (optimizations 1 + 2 + 3).

Measured with the official evaluator (`evaluate_12.py`, full 5000 steps, fresh
process per trial, 5 interleaved trials per solution on the same machine):

| Solution | Mean of 5 runs | Stdev | Per-run times (s) | Final fidelity across runs | Result |
| --- | ---: | ---: | --- | --- | --- |
| Reference | 9.020 s | 0.083 | 8.97, 9.00, 8.92, 9.11, 9.10 | 0.86855-0.87011 | 5/5 PASS |
| `solution_12_batched.py` | 2.316 s | 0.013 | 2.34, 2.31, 2.31, 2.31, 2.31 | 0.86892-0.87003 | 5/5 PASS |
| `solution_12_fused.py` | **2.128 s** | 0.019 | 2.14, 2.10, 2.13, 2.15, 2.12 | 0.86890-0.87020 | 5/5 PASS |

Mean speedup **3.90x** (batched) and **4.24x** (fused). The final-fidelity
ranges of all three solutions overlap completely; per-run values differ only
through complex64 round-off noise (see "Equivalence evidence"). Raw data:
`benchmark_results.json`.

## Where the reference actually spends its time

Stage-by-stage timing of the reference on this machine
(`profiling/profile_reference.py`, AOT-split via `jit(...).lower()` /
`.compile()`):

| Stage | Reference | Share |
| --- | ---: | ---: |
| quimb MPS -> QuOperator conversion | 0.15 s | 2% |
| jit trace (includes omeco path search) | 1.89 s | 19% |
| XLA compile (StableHLO ~8900 lines) | 3.90 s | 40% |
| 5000-step loop (0.757 ms/step) | 3.78 s | 39% |
| Total | 9.73 s | |

Two facts drive the whole optimization:

1. **~60% of the runtime is trace + XLA compile, not training.** The graph is
   huge because each of the 31 `su4` gates is built separately: 15 scalar
   multiply/stack ops plus one norm-adaptive Pade-13 `expm` per gate, times
   31, plus the overlap network and its AD transpose.
2. **Inside the step, gate construction dominates, not the contraction.** A
   step containing only the batched gate build (no network at all) costs
   0.443 ms vs 0.450 ms for a full fused step, and swapping only the `expm`
   implementation moves the gate build from 0.379 ms to 0.109 ms
   (`profiling/proto_expm.py`). The MPS-circuit overlap itself is nearly
   free: the exact circuit state has bond dimension <= 4 (exactly one
   brickwork gate crosses each cut), so all contraction tensors are tiny.

## The three changes

### 1. Batched fixed-order Pade expm for all 31 SU4 gates (both variants)

All gates are built in one shot from a `(31, 15)` parameter tensor: one
einsum against the stacked su(4) generators (same generator order as
`tc.gates.su4_gate`), then one batched matrix exponential

```text
expm(-iH) ~= (Pade33(-iH / 2^5))^(2^5)
```

with a diagonal Pade(3,3) core and a fixed 2^5 scaling. Because the argument
is anti-Hermitian, the diagonal Pade approximant is **exactly unitary** (a
Cayley-type transform); the approximation error only affects *which* unitary,
and is below the complex64 noise floor over this protocol's entire parameter
range: the largest generator spectral-norm bound observed across all 5000
training steps is 3.60 (`profiling/diag_equivalence.py`), i.e. 0.113 after
scaling, where the measured deviation from a float64 `expm` is ~3e-6 and the
approximant stays float64-accurate up to ||H|| ~ 11. Unlike the generic
norm-adaptive Pade-13 `expm`, the graph is static (no norm estimate, no
branches), so it traces, compiles, and differentiates cheaply. This single
change is worth about 3x end to end.

### 2. Qubit-pair fusion: depth-2 brickwork -> 16-site depth-1 chain (fused variant)

Adjacent qubit pairs `(2j, 2j+1)` are treated as one four-level site
(`tc.QuditCircuit(16, dim=4)`):

- layer-1 SU4 gates act *inside* a pair -> single-site 4x4 unitaries, with
  the Neel `|01>` preparation folded in as a constant basis permutation;
- layer-2 SU4 gates straddle two fused sites -> two-site 16x16 unitaries
  `I2 (x) SU4 (x) I2` (one batched einsum);
- the DMRG target MPS is pair-fused once outside the training loop
  (16 tensors, chi <= 8, physical dimension 4) and wrapped as a
  `tc.quantum.QuVector` bra.

The overlap is still computed by the framework as the direct tensor-network
contraction `(target_bra @ circuit.quvector()).eval()`; every contraction
result is unchanged, but the network has half the nodes. After optimization 1
the contraction is no longer the bottleneck, so this contributes only the
final ~0.19 s (mostly faster tracing): 2.32 s -> 2.13 s.

### 3. `jax.lax.scan` over the 5000 Adam updates (both variants)

The update step is compiled exactly once and the loop runs inside XLA with no
per-step Python dispatch, returning the loss/fidelity/overlap histories as
stacked scan outputs. Measured alone on the unmodified reference objective
this is only a ~9% end-to-end win (8.26 s vs 9.02 s: per-call dispatch is
just ~5 us and the compile cost stays), which is why the graph-level changes
above matter more.

Optimized stage profile of the fused variant (same AOT split as above):

| Stage | Fused variant | vs reference |
| --- | ---: | ---: |
| target pair-fusion + wrap | 0.12 s | 1.3x |
| jit trace | 0.52 s | 3.6x |
| XLA compile (StableHLO 780 lines) | 0.39 s | 10.0x |
| 5000-step loop (0.198 ms/step) | 0.99 s | 3.8x |
| Total | 2.02 s (2.13 s via evaluator, cold) | 4.5x |

## Equivalence evidence

- **Same objective:** on random parameter vectors the restructured objectives
  match the reference objective's complex overlap to ~1e-12
  (`profiling/proto_variants.py`).
- **Same trajectory:** running fused and reference from the identical seed,
  the first 10 recorded losses agree bit-for-bit in float32; the deviation is
  3.5e-5 after 50 steps and 1.3e-3 after 400 steps
  (`profiling/diag_equivalence.py`), i.e. ordinary round-off noise amplified
  by optimizer dynamics -- the same magnitude of run-to-run scatter the
  reference itself shows across its own 5 benchmark trials (final fidelity
  0.86855-0.87011).
- **Same verdicts:** all 15 benchmark runs print `Overall: PASS`;
  `static_policy.py` scores both optimized files 1.0 (100 and 129 effective
  lines, no raw-simulator or cheating hits).

## What did not work (measured dead ends)

| Attempt | Result | Why |
| --- | --- | --- |
| `lax.scan` alone on the reference | 8.26 s (-9%) | dispatch is only ~5 us/step; compile unchanged |
| Batched gates via generic `jexpm`, no scan | 4.28 s | compile drops 6x but per-step still expm-bound |
| Fusing 4 qubits per site (dim=16) | 16.2 s | 256x256 crossing gates blow up contraction FLOPs |
| Absorbing single-site gates into two-site gates | no gain | absorption einsums cost what the removed nodes saved |
| `lax.scan` with `unroll=4` | worse | larger compile, slower steps |
| `--xla_cpu_multi_thread_eigen=false`, core pinning | no gain / worse | kernels too small to parallelize anyway |

## Environment and reproduction

- Cloud VM, 4 vCPU Intel Xeon x86_64, 15 GB RAM, Linux 6.12.
- Python 3.12.3 venv with the benchmark-pinned dependency set from
  `frameworks/tensorcircuit/requirements.txt`: tensorcircuit-nightly
  1.8.0.dev20260726, jax/jaxlib 0.10.0, optax 0.2.8, quimb 1.11.1,
  numpy 2.2.6, omeco 0.2.4 (no Docker available on this VM; all solutions
  ran interleaved in the identical environment, so the ratios follow the
  repository's same-machine/same-environment T/T_ref convention). The
  reference's 9.02 s mean here is consistent with the 9.73 s recorded for it
  in the canonical py311 image on the same class of VM
  (`results/fable5/challenge-12/`).

```bash
python3 -m venv --without-pip .venv-c12 && \
  curl -sS https://bootstrap.pypa.io/get-pip.py | .venv-c12/bin/python - && \
  .venv-c12/bin/pip install -r frameworks/tensorcircuit/requirements.txt
cd optimized_sloutions/challenge-12
/path/to/.venv-c12/bin/python benchmark_runner.py 5
```

## Upstream porting notes

Both files are drop-in replacements for the challenge-suite reference
`solution_12.py`: same `run_solution(config)` contract, same returned keys
and shapes, no new dependencies. For an upstream PR the recommended primary
candidate is `solution_12_batched.py`: it is a small, easily reviewed diff
against the current reference (the circuit-building code is untouched), works
for any brickwork layout, and captures 3.90x of the 4.24x. The pair-fused
variant documents where the remaining ~8% lives (`tc.QuditCircuit`,
`tc.quantum.QuVector`, and the omeco contractor it uses are already exercised
elsewhere in the suite, but the construction is specific to the fixed
two-layer even-qubit brickwork of this problem).
