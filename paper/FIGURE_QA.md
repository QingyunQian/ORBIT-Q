# Figure QA record

Date: 2026-08-03

- The two latest benchmark figures are byte-identical to the PDF/PNG/SVG
  assets in `origin/codex/deepseek-v4-flash-high-benchmark` (upstream PR #23).
- All 12 expert `factor-ablation.svg` files are byte-identical to their source
  Task branches; Task 05 is sourced from `codex/task-05-tc-native-fused`.
- The benchmark source table contains 5 configurations and the task table
  contains exactly 60 rows (5 × 12).
- Final pass counts cross-check against the task table: 10, 10, 9, 9, and 5.
- Task 08 is final `F` for all five configurations.
- The outcome and resource PNGs were inspected at full resolution; labels are
  inside the canvas and the selected-model set contains neither Fable 5 nor
  DeepSeek max.

Because the delivered figures are reused upstream assets rather than newly
drawn panels, no additional plotting backend or regenerated visual output is
introduced in this branch.
