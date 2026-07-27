# Workload dataset policy

The workload dataset is a required gate for autoresearch. The repository does not contain a completed dataset. `datasets/public/manifest.json` has `status: "not_built"` and no cases.

## Tiers

### Public development

Proposal agents may read public cases. Include, for each task:

- the canonical evaluator configuration;
- a reduced smoke case where the task contract permits one;
- a scale probe that preserves the scientific method and output contract.

If a task forbids configuration changes, use approved public seeds or inputs that exercise the same contract.

Store public records under `datasets/public/`. List each record in `datasets/public/manifest.json`.

### Hidden rotating tuning

The trusted controller owns hidden tuning records. Provide at least two disjoint rotations for each task. Rotate after exposure, controller compromise, or a declared research cycle.

Do not store hidden records, manifests, paths, seeds, hashes, keys, or populated controller configuration in the Git checkout. The controller returns aggregate validity and runtime fields.

### Sealed final holdout

The trusted controller stores the holdout apart from hidden tuning data and credentials. Proposal agents cannot query it.

Evaluate a promoted candidate once per holdout version. Return aggregate validity and runtime fields. Do not return case-level output, timing, or failure details. A team that tunes after a holdout result must create a new holdout version before making another final claim.

## Record requirements

Each workload record must define:

- dataset version;
- tier;
- task ID and case ID;
- input configuration and seed;
- expected result keys, shapes, and numerical meaning;
- functional validity thresholds;
- source provenance;
- content SHA256.

Hidden records use the same internal schema. Do not publish their values or identifiers.

Download external inputs from their primary source when a task needs them. Record the source URL, retrieval date, license, upstream version, and downloaded content hash.

## Coverage and separation

Cover challenges 01 through 12 in all three tiers. Attempt each record with the
matching human expert solution in the pinned environment. If that immutable
expert has a reproducible setup failure, retain the failure and use an
independent trusted oracle to validate the record's scientific semantics before
marking the dataset ready. This exception does not create a runtime baseline.

Reject a dataset release when:

- an expert fails a record;
- public, tuning, and holdout records share a config and seed;
- a record changes the task's scientific semantics;
- a task lacks required coverage;
- a record lacks provenance or a content hash.

## Versioning

Assign immutable versions in this form:

```text
orbitq-workloads-vYYYYMMDD.N
```

Create a new version when any record, threshold, seed, generator, or split changes. Do not overwrite a released version. Record the generator commit and environment image digest.

## Build and release procedure

1. Read every task problem statement and evaluator.
2. Define public, tuning, and holdout coverage for all 12 tasks.
3. Generate records with deterministic tools and fixed seeds.
4. Run each human expert against each record.
5. Review failures and semantic changes.
6. Compute record and manifest hashes.
7. Place public records in `datasets/public/`.
8. Move private records and keys to controller-owned storage outside Git.
9. Assign the dataset version.
10. Set the public manifest status to `ready`.
11. Have the trusted controller attest that tuning rotations and the sealed holdout exist and pass expert validation.

Do not set the manifest to `ready` before these steps finish.

## Proposal-agent boundary

A proposal agent may receive:

- public records and their manifest;
- aggregate hidden validity;
- aggregate hidden mean and median runtime;
- aggregate runtime standard error, paired percentage improvement, and paired
  speedup with standard errors;
- timeout state and passing-run count.

A proposal agent must not receive:

- hidden records or record identifiers;
- private paths, filenames, hashes, seeds, keys, tokens, or credentials;
- per-record hidden outputs, errors, thresholds, timings, or pass states;
- controller logs, stack traces, commands, environment dumps, or process listings that expose private storage.

The controller must redact private values before it writes reports or returns errors.

## Private configuration

`private-data.example.toml` documents controller settings with placeholder values. Copy it to controller-owned storage outside any Git checkout. Never commit a populated copy. Never mount a populated copy into a proposal-agent worktree.
