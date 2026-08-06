# Figure QA record

Date: 2026-08-06

- Four figures were regenerated from tracked CSV sources with matplotlib.
- Each figure was exported as a vector PDF manuscript master and a matched
  300-dpi PNG used only for inline GitHub/PR preview.
- SVG and TIFF are intentionally omitted because every panel is line art and
  PDF remains the publication master.
- The generic figure preflight therefore reports missing SVG/TIFF exports;
  these are accepted format-contract exceptions, not missing deliverables.
- PDF output uses embedded TrueType fonts and contains no raster image object.
- Temporary Python-rendered QA previews were inspected at full resolution for
  clipped labels, overlap, stretched elements, and panel consistency.
- Final pass counts cross-check to **10, 10, 9, 9, 5, 8**; Task 08 is failed
  for all six added configurations.
- Fig. 2b runtime ratios were recomputed against the original paper's public
  expert TensorCircuit runtimes and include passed tasks only.
- Fig. 2b preserves the original paper's reference marker and legacy model
  palette/label anchors; its displayed ranges are compacted to the occupied
  data region, added-campaign labels are separated, and Sol uses a
  non-blue color distinct from GPT-5.5.
- The expert overview cross-checks all 12 end-to-end values against the
  corresponding upstream Task PR; Task 05 uses legal PR #19 and Task 08 is
  included as a valid optimized implementation.
- All scatter points use circles and direct labels without colored leader
  lines. Fig. 4b/e use one total-token encoding across configurations.
- Fig. 1c has no old/new separator and no repeated `high` qualifier in model
  labels. The effort convention is stated once in prose.
- Grok 4.5 appears in the outcome, runtime, wall-time, token, and cost views.
  Its Fig. 4c cost is reconstructed from recorded token classes at xAI public
  list prices and cross-checks to $5.090140 total / $0.636267 per valid result.
- Neither Fable 5 nor DeepSeek max appears in the updated figures.
