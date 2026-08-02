# Figure QA record

Date: 2026-08-03

- Four figures were regenerated from tracked CSV sources with matplotlib.
- Each figure was exported as editable SVG, vector PDF, and 320 dpi PNG.
- SVG/PDF are the submission deliverables; TIFF is intentionally omitted
  because every panel is line art and the requested master files are vector.
- SVG output retains text as text and contains no embedded raster image.
- PDF output uses embedded TrueType fonts and contains no raster image object.
- PNG previews were inspected at full resolution for clipped labels, overlap,
  stretched elements, and panel consistency.
- Final pass counts cross-check to **10, 10, 9, 9, 5**; Task 08 is failed for
  all five new configurations.
- Fig. 2b runtime ratios were recomputed against the original paper's public
  expert TensorCircuit runtimes and include passed tasks only.
- The expert overview cross-checks all 12 end-to-end values against the
  corresponding upstream Task PR; Task 05 uses legal PR #19 and Task 08 is
  included as a valid optimized implementation.
- All scatter points use circles and direct labels without colored leader
  lines. Fig. 4b/e use one total-token encoding across configurations.
- Neither Fable 5 nor DeepSeek max appears in the updated figures.
