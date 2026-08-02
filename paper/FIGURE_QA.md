# Figure QA record

Date: 2026-08-03

- Four figures were regenerated from tracked CSV sources with matplotlib.
- Each figure was exported as vector PDF only.
- SVG, PNG, and TIFF are intentionally omitted because the requested master
  files are PDF and every panel is line art.
- The generic figure preflight therefore reports missing SVG/raster exports;
  these are accepted format-contract exceptions, not missing deliverables.
- PDF output uses embedded TrueType fonts and contains no raster image object.
- Temporary Python-rendered QA previews were inspected at full resolution for
  clipped labels, overlap, stretched elements, and panel consistency.
- Final pass counts cross-check to **10, 10, 9, 9, 5**; Task 08 is failed for
  all five new configurations.
- Fig. 2b runtime ratios were recomputed against the original paper's public
  expert TensorCircuit runtimes and include passed tasks only.
- Fig. 2b preserves the original paper's axes, reference marker, and legacy
  model palette/label anchors; only the added-campaign labels were separated
  to prevent overlap, with Sol high assigned a non-blue color distinct from
  GPT-5.5.
- The expert overview cross-checks all 12 end-to-end values against the
  corresponding upstream Task PR; Task 05 uses legal PR #19 and Task 08 is
  included as a valid optimized implementation.
- All scatter points use circles and direct labels without colored leader
  lines. Fig. 4b/e use one total-token encoding across configurations.
- Neither Fable 5 nor DeepSeek max appears in the updated figures.
