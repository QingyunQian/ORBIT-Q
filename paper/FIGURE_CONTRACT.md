# Figure contract

The figures in this directory are intended for paper drafting and follow the
separation used by ORBIT-Q's published views:

1. **Validity layer:** final pass/fail counts and the task-level matrix.
2. **Artifact layer:** human-expert baseline/optimized runtime and speedup.
3. **Agent-resource layer:** wall time, solving-side tokens, and cost per valid
   solution for comparable Docker-agent runs.

No quantity from one layer is used as a proxy for another. In particular,
agent wall time is not artifact runtime, and a missing expert ratio is not
imputed. Fable is shown in the validity layer but omitted from comparable
agent-resource plots.

## Export requirements

- Python/matplotlib backend only.
- Editable SVG and PDF text (`svg.fonttype=none`, `pdf.fonttype=42`).
- 600-dpi PNG and TIFF raster exports.
- Width approximately 181.6 mm (7.15 in), suitable for a two-column figure
  or a full-width draft figure.
- Minimum explicit text size 5 pt; the current script uses 5.7 pt or larger.
- Colorblind-safe qualitative palette; no rainbow colormap.
- Log axes have positive measured values and are labeled as log scale.

## Source and regeneration

`figures/make_paper_figures.py` is the sole figure generator. It reads the
four CSVs in `source_data/`, writes all five figure families, and is checked by
the Nature-figure static preflight plus rendered visual inspection.
