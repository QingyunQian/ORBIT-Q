# ORBIT-Q paper summary

这是当前结果的 paper-facing 简明汇总。图形版式参考了
[ORBIT-Q README](https://github.com/sxzgroup/ORBIT-Q) 和
[论文 arXiv:2607.03105](https://arxiv.org/abs/2607.03105)。

## 最新结果图（直接使用）

最重要的是下面这张总图：

**[Figure 1 — 最新 benchmark 总结果（SVG）](figures/fig1_dual_axis_updated.svg)**
[PDF](figures/fig1_dual_axis_updated.pdf) · [PNG](figures/fig1_dual_axis_updated.png) · [TIFF](figures/fig1_dual_axis_updated.tiff)

配套图：

- [Figure 2 — 12 个 Task 的 P/F 矩阵（SVG）](figures/fig2_task_level_matrix.svg)
- [Figure 3 — Agent wall time / token / cost（SVG）](figures/fig3_agent_resources.svg)
- [Figure 4 — 人类专家优化前后运行时间（SVG）](figures/fig4_expert_speedups.svg)
- [Figure 5 — 主要优化 insight 与加速倍数（SVG）](figures/fig5_insight_speedups.svg)

每张图同时提供 PDF、PNG、600-dpi TIFF。图形生成脚本是
[`figures/make_paper_figures.py`](figures/make_paper_figures.py)。

## 最新 benchmark 结果

| Solver | Effort | Final valid |
|---|---:|---:|
| GPT-5.6 Sol | high | **10/12** |
| GPT-5.6 Sol | ultra | **10/12** |
| GPT-5.6 Terra | high | **9/12** |
| GPT-5.6 Luna | high | **9/12** |
| DeepSeek V4 Flash | high | **5/12** |
| DeepSeek V4 Flash | max | **5/12** |
| Fable 5 | reported | **12/12** |

Fable 5 是独立 Cursor/Fable 流程，只保留其 artifact 结果，不与 Docker
Agent 的 token、wall time、cost 做横向比较。

Task 08 按最终人类专家复核统一记为失败；Luna 和 Sol ultra 的原始
verifier reward 保留在数据表中，但不计入最终 paper-valid 数字。

完整 task 矩阵：[`source_data/task_outcomes.csv`](source_data/task_outcomes.csv)。

## 人类专家优化的核心结论

| Task | 主要方法 | End-to-end speedup |
|---:|---|---:|
| 07 | classical ancilla / duplicate trajectory reduction | **45.758×** |
| 05 | exact bounded-rank no-QR MPS | **14.076×** |
| 01 | batched closed-form gate construction | **9.636×** |
| 10 | native hyperedge + fixed contraction path | **4.898×** |
| 03 | TensorCircuit `K.vmap` local maps | **4.894×** |
| 12 | batched fixed-order Padé SU(4) | **3.914×** |
| 09 | causal-cone pruning + compiled trajectory | **3.822×** |

最重要的两个 loophole 是：

1. **Task 03:** 每层 post-selection 使问题精确退化为 6 条独立单比特轨迹。
2. **Task 07:** ancilla 只产生两个不同的 classical branches，可精确合并重复轨迹。

Task 10 的优势主要来自 cold specialization 与固定 contraction path；并非
MPO 在门级计算上天然优于 hyperedge。Task 08 的 1.045× 区间跨过 1×，不
报告为确认的性能提升。

完整 12-task 表：[`source_data/expert_optimization.csv`](source_data/expert_optimization.csv)。

## 如何复现图

```bash
MPLCONFIGDIR=/tmp/orbitq-mpl python3 paper/figures/make_paper_figures.py
python3 /Users/qqy/.codex/skills/nature-figure/scripts/validate_figure.py \
  paper/figures/make_paper_figures.py
```

数据口径见 [`source_data/README.md`](source_data/README.md)，图形规范见
[`FIGURE_CONTRACT.md`](FIGURE_CONTRACT.md)，QA 记录见
[`FIGURE_QA.md`](FIGURE_QA.md)。
