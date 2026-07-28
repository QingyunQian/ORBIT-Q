# Task-scoped research records

Keep every campaign's tracked research records under:

```text
research/task-XX/
├── LOG.md
├── INSIGHTS.md
├── SURVEY.md
├── IMPLEMENTATION_COMPARISON.md
└── profiles/
```

The files have different roles:

- `LOG.md` is the append-only evidence ledger. Record every run, failure,
  timeout, report hash, and decision in chronological order.
- `INSIGHTS.md` is the maintained synthesis. Consolidate what was learned,
  which ideas remain promising, and which approaches should not be repeated
  without new evidence.
- `SURVEY.md` freezes the cited technical survey, semantic constraints,
  hypotheses, and measurement rule before optimization.
- `IMPLEMENTATION_COMPARISON.md` is the campaign closeout report for the
  accepted implementation and its eligible evidence.
- `profiles/` holds sanitized, task-specific profiler outputs.

Task-specific profiling or analysis scripts may live beside these records.
Machine-local raw reports, `research/task-XX/results.tsv`, and
`research/task-XX/run.log` remain untracked and must be archived according to
`GOAL.md`.

Create `LOG.md` and `INSIGHTS.md` from the templates under `autoresearch/`.
Do not use a root-level `LOG.md`: it mixes independent campaigns and makes
evidence ownership ambiguous.
