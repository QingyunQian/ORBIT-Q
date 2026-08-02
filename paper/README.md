# ORBIT-Q: updated results for the paper

This note contains only the five new solver configurations requested for the
paper: Sol high, Sol ultra, Terra high, Luna high, and DeepSeek V4 Flash high.
Fable 5 and DeepSeek max are not included.

## Latest result figures

### Task-level outcomes

[SVG](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-outcomes.svg) ·
[PDF](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-outcomes.pdf) ·
[PNG](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-outcomes.png)

![Final outcomes for the five solver configurations](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-outcomes.png)

### Agent-side resource use

[SVG](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-agent-resource-use.svg) ·
[PDF](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-agent-resource-use.pdf) ·
[PNG](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-agent-resource-use.png)

![Agent wall time, token use, and cost per valid solution](../results/deepseek-v4-flash-high/figs/deepseek-v4-flash-high-agent-resource-use.png)

These are the canonical figures from
[ORBIT-Q PR #23](https://github.com/sxzgroup/ORBIT-Q/pull/23). The resource
figure keeps the paper's wall-time / token / efficiency panel structure. The
original paper figures remain unchanged in [`docs/assets`](../docs/assets/).

## Final results

| Solver | Effort | Valid tasks |
|---|---:|---:|
| GPT-5.6 Sol | high | **10/12** |
| GPT-5.6 Sol | ultra | **10/12** |
| GPT-5.6 Terra | high | **9/12** |
| GPT-5.6 Luna | high | **9/12** |
| DeepSeek V4 Flash | high | **5/12** |

Task 08 is `F` for all five configurations after final human-expert review.
The raw Luna and Sol-ultra verifier rewards are retained in
[`task_outcomes.csv`](source_data/task_outcomes.csv), but are not counted as
paper-valid results.

## Human-expert optimization figures

The figures below are copied unchanged from the corresponding upstream PRs.
Each is a direct factor-removal/ablation plot; factors from different panels
are not additive.

| Task | End-to-end result | Upstream PR | Factor-ablation figure |
|---:|---:|---:|---|
| 01 | **9.636×** | [#8](https://github.com/sxzgroup/ORBIT-Q/pull/8) | [SVG](../optimized_solutions/challenge-01/factor-ablation.svg) |
| 02 | **1.116×** | [#9](https://github.com/sxzgroup/ORBIT-Q/pull/9) | [SVG](../optimized_solutions/challenge-02/factor-ablation.svg) |
| 03 | **4.894×** | [#10](https://github.com/sxzgroup/ORBIT-Q/pull/10) | [SVG](../optimized_solutions/challenge-03/factor-ablation.svg) |
| 04 | **2.602×** | [#11](https://github.com/sxzgroup/ORBIT-Q/pull/11) | [SVG](../optimized_solutions/challenge-04/factor-ablation.svg) |
| 05 | **1.939× mean paired** | [#19](https://github.com/sxzgroup/ORBIT-Q/pull/19) | [SVG](../optimized_solutions/challenge-05/factor-ablation.svg) |
| 06 | **1.504×** | [#13](https://github.com/sxzgroup/ORBIT-Q/pull/13) | [SVG](../optimized_solutions/challenge-06/factor-ablation.svg) |
| 07 | **45.758×** | [#7](https://github.com/sxzgroup/ORBIT-Q/pull/7) | [SVG](../optimized_solutions/challenge-07/factor-ablation.svg) |
| 08 | **1.045×; not confirmed** | [#18](https://github.com/sxzgroup/ORBIT-Q/pull/18) | [SVG](../optimized_solutions/challenge-08/factor-ablation.svg) |
| 09 | **3.822×** | [#14](https://github.com/sxzgroup/ORBIT-Q/pull/14) | [SVG](../optimized_solutions/challenge-09/factor-ablation.svg) |
| 10 | **4.898×** | [#15](https://github.com/sxzgroup/ORBIT-Q/pull/15) | [SVG](../optimized_solutions/challenge-10/factor-ablation.svg) |
| 11 | **1.464×** | [#16](https://github.com/sxzgroup/ORBIT-Q/pull/16) | [SVG](../optimized_solutions/challenge-11/factor-ablation.svg) |
| 12 | **3.914×** | [#17](https://github.com/sxzgroup/ORBIT-Q/pull/17) | [SVG](../optimized_solutions/challenge-12/factor-ablation.svg) |

The strongest findings are the exact Task 07 classical-ancilla reduction,
the Task 03 product-state reduction, and Task 10 cold contraction-program
specialization. For Task 05, the paper-facing result is the legal
TensorCircuit-native PR #19 result: OMECo path-search budget is the dominant
measured factor; the earlier 14.08× custom-MPS result is not used.

Detailed source values and review overrides are in
[`source_data/`](source_data/). The original paper and layout reference are
[arXiv:2607.03105](https://arxiv.org/abs/2607.03105) and the
[ORBIT-Q repository](https://github.com/sxzgroup/ORBIT-Q).
