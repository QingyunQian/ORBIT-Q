# Task 08 — bounded TensorCircuit sampling batches

## Take-home insight

**Keep the public expert’s TensorCircuit circuit and all 8192
`perfect_sampling` trajectories, but execute the mapped sampler in contiguous
256-shot blocks.** The expert’s single 8192-shot `vmap` puts the shot axis into
every XLA contraction intermediate; bounding that axis is the only material
change and converts the canonical 7-GiB workload from **0/5 OOM** to
**5/5 PASS**.

## Factor attribution

| Factor | Canonical/full-workload result | Decision |
| --- | ---: | --- |
| One 8192-shot mapped batch | 0/5 PASS; attempted buffers 9.31–17.97 GiB | Root cause |
| 512-shot blocks | 50.843 s single screen | Feasible, not best |
| **256-shot blocks** | **44.028 s single screen** | **Promote** |
| 128-shot blocks | 55.182 s single screen | Dispatch overhead |

![Task 08 factor attribution](factor-ablation.svg)

*Figure — The bounded 256-shot execution is the necessary OOM-to-PASS factor
and the best of the three feasible chunk-size screens. Chunk-size timings are
single full-workload screens, not uncertainty estimates.*

## What the factors mean

- **Bounded mapped batch:** the same cached TensorCircuit
  `K.jit(K.vmap(perfect_sampling))` function consumes 32 contiguous slices
  instead of the complete status matrix at once. Every seeded status row is
  still consumed exactly once and in the original order.
- **Chunk size:** 512 leaves larger contraction intermediates live; 128 adds
  excess host dispatch. A 256-shot block is the measured balance for this
  workload.

## Result and claim boundary

At 7 GiB, the immutable expert failed all five canonical attempts, while the
candidate passed all five with a mean runtime of **47.356 s**. This establishes
a peak-memory and reproducibility improvement; it does not provide a numerical
speedup denominator.

With sufficient memory, the expert is algorithmically runnable. A separate
64-GiB, six-CPU-affinity five-pair session produced 126.675 s expert and
123.188 s candidate means. The candidate won only 3/5 pairs and its mean
pairwise speedup was 1.045x with 95% t-CI [0.818, 1.273]. Therefore this PR
does **not** claim a confirmed runtime speedup.
