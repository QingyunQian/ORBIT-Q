# Figure QA record

Date: 2026-08-03  
Backend: Python / matplotlib  
Generator: `paper/figures/make_paper_figures.py`

## Automated checks

The Nature-figure static preflight completed with **14 pass, 0 warn, 0 fail**:

```text
python3 /Users/qqy/.codex/skills/nature-figure/scripts/validate_figure.py \
  paper/figures/make_paper_figures.py
```

The generator was then executed with:

```text
MPLCONFIGDIR=/tmp/orbitq-mpl python3 paper/figures/make_paper_figures.py
```

All five figure families contain SVG, PDF, PNG, and TIFF outputs. The raster
files are exported at 600 dpi and the vector files retain editable text.

## Visual checks

- Figure 1: pass counts are labeled directly; missing artifact/reference
  ratios are stated in the panel rather than plotted as invented values.
- Figure 2: all 84 model/task cells are populated; Task 08 is visibly `F` for
  the six GPT-5.6/DeepSeek rows after final review.
- Figure 3: Fable is omitted from the resource axis; DeepSeek high/max labels
  are vertically separated to remain legible at paper scale.
- Figure 4: baseline and optimized runtimes share task order; log scales are
  labeled; Task 08 is called out as not a confirmed speedup.
- Figure 5: the two exact design reductions are colored separately from the
  framework-native implementation factors; bars are explicitly non-additive.

The figure contract is in [`FIGURE_CONTRACT.md`](FIGURE_CONTRACT.md). A
manuscript should still check final placement, caption width, and journal font
embedding after the figures are imported into the target TeX/Typst document.
