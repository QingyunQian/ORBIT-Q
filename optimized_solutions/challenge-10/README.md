# Challenge 10: exact low-rank contraction

**Take-home insight.** The `4.90x` speedup comes from keeping the complete VQE
calculation in its exact low-bond-dimension MPS form and applying local
operators directly. It does **not** come from an intrinsically cheaper MPO
replacement for TensorCircuit-NG's native CMZ hyperedge.

## Result and attribution

| Change or ablation | Five-pair result | Interpretation |
|---|---:|---|
| Public expert → exact low-rank solver | **`4.898x`** (`4.598–5.199x`) | Dominant algorithmic gain |
| Generic hyperedge → generic MPO, normal preprocessing | `1.535x` (`1.505–1.566x`) | MPO appears faster |
| Same generic comparison, preprocessing disabled | `0.963x` (`0.899–1.027x`) | No resolved gate-level advantage |
| Hyperedge → MPO inside the exact low-rank solver | `1.000x` (`0.977–1.023x`) | End-to-end neutral |

Intervals are 95% paired t-intervals. Values above one favor the method after
the arrow.

![Task 10 CMZ representation ablation](factor-ablation.svg)

*Figure — **a**, the generic MPO wins only when TensorCircuit's preprocessing
is available; with preprocessing matched, and inside the optimized solver,
hyperedge and MPO are indistinguishable. **b**, the reason is structural:
the native CMZ's `CopyNode`s prevent the current single-gate merge pass, whereas
the ordinary MPO is reduced from 424 contraction tensors to 86.*

## Why the apparently cheaper MPO wins

TensorCircuit-NG's built-in
[`cmz_gate`](https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/tensorcircuit/gates.py#L1051)
is already an exact bond-2 diagonal MPS connected through `CopyNode`
hyperedges. The representation is not computationally inferior. However, the
current
[contractor preprocessing](https://github.com/tensorcircuit/tensorcircuit-ng/blob/master/tensorcircuit/cons.py#L1035-L1038)
skips `_merge_single_gates` whenever a `CopyNode` is present.

That implementation detail changes the generic network:

| Generic contraction | Contracted tensors | Path search |
|---|---:|---:|
| Native hyperedge | 424 | 3.54 s |
| MPO with preprocessing | 86 | 0.73 s |
| MPO without preprocessing | 424 | 3.72 s |

Accordingly, enabling preprocessing for the generic MPO is itself a `1.56x`
end-to-end improvement. Once that advantage is removed, the hyperedge is
nominally faster, but the five-pair interval includes equality.

## Correctness and final decision

The low-rank MPO and hyperedge variants produced bitwise-identical initial
energy and gradient and the same final energy (`-1.1781773567`) in every pair.
The submitted solution therefore keeps the deterministic low-rank algorithm;
its local MPO is an implementation choice, not the source of the headline
gain. A framework-level opportunity would be a hyperedge-aware gate-fusion
pass that preserves `CopyNode` semantics.

The original expert-versus-optimized comparison used the same
network-disabled 6-CPU/7-GiB container and TensorCircuit-NG
`1.8.0.dev20260726`: `18.931296 s` versus `3.869287 s` over five
counterbalanced pairs.
