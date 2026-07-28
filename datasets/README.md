# Workload dataset policy

The public workload dataset is a required gate for autoresearch. The repository
does not contain a completed dataset. `datasets/public/manifest.json` has
`status: "not_built"` and no cases.

## Campaign scope

Build the dataset for the one task selected by the campaign. A campaign does
not need workloads for the other 11 tasks.

All workload records, configurations, seeds, evaluators, and validity rules are
public and versioned. Hidden tuning sets, sealed holdouts, private
configuration, and controller attestations are not required.

## Public workload

Include for the selected task:

- the canonical evaluator configuration;
- a reduced smoke case when the task contract permits one;
- a scale probe when the task contract permits one.

If the selected task has one fixed deterministic configuration and no seed,
the canonical case alone can satisfy the gate. Do not invent rotations,
configurations, or seeds that change the task contract merely to create more
cases.

Store records under `datasets/public/` and list each one in
`datasets/public/manifest.json`.

## Record requirements

Each workload record must define:

- dataset version;
- task ID and case ID;
- input configuration and any applicable public seed;
- expected result keys, shapes, and numerical meaning;
- functional validity thresholds;
- source provenance;
- content SHA256.

Download external inputs from their primary source when a task needs them.
Record the source URL, retrieval date, license, upstream version, and
downloaded content hash.

## Validation

Attempt each selected-task record with the matching human expert solution in
the pinned environment. If that immutable expert has a reproducible setup
failure, retain the failure and use an independent trusted oracle to validate
the record's scientific semantics before marking the dataset ready. This
exception does not create a runtime baseline.

Reject a dataset release when:

- an expert fails a record;
- a record changes the task's scientific semantics;
- the selected task lacks required coverage;
- a record lacks provenance or a content hash.

## Versioning

Assign immutable versions in this form:

```text
orbitq-workloads-vYYYYMMDD.N
```

Create a new version when any record, threshold, seed, generator, or split changes. Do not overwrite a released version. Record the generator commit and environment image digest.

## Build and release procedure

1. Read the selected task's problem statement and evaluator.
2. Define public coverage without changing the task contract.
3. Generate any records with deterministic tools and public seeds.
4. Run the human expert against each record.
5. Review failures and semantic changes.
6. Compute record and manifest hashes.
7. Place records in `datasets/public/`.
8. Assign the dataset version and selected task ID.
9. Set the public manifest status to `ready`.

Do not set the manifest to `ready` before these steps finish.

## Benchmark reporting

Record the public workload version, manifest hash, validity result, evaluator
runtime, paired statistics, and immutable report hash for every experiment.
Because the workload is public, claims are limited to that workload and task.

Run a promoted candidate once more as a fresh final paired benchmark on the
immutable public workload. If tuning continues afterward, record the later
candidate as a new experiment and rerun the final paired benchmark before
making a claim.
