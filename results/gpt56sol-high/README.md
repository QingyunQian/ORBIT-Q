# GPT-5.6 Sol High — TensorCircuit Benchmark

This directory contains one Harbor trial for each of the 12 ORBIT-Q
TensorCircuit challenges.

## Protocol

- Run date: 2026-07-28
- Base commit: `0201238ec2983907e2891f5319f5fff2d00844d5`
- Solver agent: Codex
- Solver model: `gpt-5.6-sol`
- Solver reasoning effort: `high`
- Audit model: `gpt-5.6-sol`
- Audit reasoning effort: `high`
- Framework: TensorCircuit-NG / `tensorcircuit-nightly`
- Docker image: `challenge-benchmark-quantum-tensorcircuit:py311`
- Trials per challenge: 1
- Harbor retries: 0

The solver and audit effort are both fixed at `high` to match the original
GPT-5.5 benchmark effort. Challenges were run sequentially.

Canonical files under `tasks/` were not changed. The Mac runner used temporary
task copies with only the resource declaration adapted to 6 CPUs, 10,240 MiB
memory, and 16,384 MiB storage so Harbor could run against the local Colima
backend. This resource adaptation can affect wall-clock runtime comparisons.

## Results

| Challenge | Reward | Functional | Static policy | LLM audit | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| 01 | 0.0 | 1.0 | 1.0 | 0.0 | 107.51 |
| 02 | 1.0 | 1.0 | 1.0 | 1.0 | 10.65 |
| 03 | 1.0 | 1.0 | 1.0 | 1.0 | 9.84 |
| 04 | 1.0 | 1.0 | 1.0 | 1.0 | 14.14 |
| 05 | 0.0 | 1.0 | 1.0 | 0.0 | 124.92 |
| 06 | 1.0 | 1.0 | 1.0 | 1.0 | 23.39 |
| 07 | 1.0 | 1.0 | 1.0 | 1.0 | 155.52 |
| 08 | 0.0 | 1.0 | 1.0 | 0.0 | 55.48 |
| 09 | 1.0 | 1.0 | 1.0 | 1.0 | 87.31 |
| 10 | 1.0 | 1.0 | 1.0 | 1.0 | 68.68 |
| 11 | 1.0 | 1.0 | 1.0 | 1.0 | 100.41 |
| 12 | 1.0 | 1.0 | 1.0 | 1.0 | 12.85 |
| **Total / mean** | **9 / 12** | **12 / 12** | **12 / 12** | **9 / 12** | **64.23 mean** |

Measured solution runtime totals 770.70 seconds. Runtime is reported separately
and does not reduce the pass reward.

## Audit failures

- Challenge 01: the candidate applied an MPS bond-dimension cap after each
  two-qubit gate. The audit considered this unrequested truncation a change to
  the specified unitary circuit.
- Challenge 05: TensorCircuit `exp1` uses
  `exp(-i * theta * U / 2)`, but the candidate omitted the required factor of
  two in the imaginary angles.
- Challenge 08: the candidate drove conditional sampling with a fixed scrambled
  Sobol sequence. The audit rejected these correlated quasi-Monte Carlo rows as
  a substitute for independent circuit shots and as evaluator exploitation.

The full audit rationale is retained in each challenge's
`audit-details.json`.

## Solver timeouts

Challenges 01 and 06 reached Harbor's solver time limit after writing their
candidate files. Harbor still ran the verifier on those candidates. Challenge
01 received reward 0.0 due to the audit failure above; challenge 06 passed all
three reward components and received reward 1.0.

## Aggregate usage

- Input tokens: 25,908,116
- Cached input tokens: 24,198,656
- Output tokens: 162,693
- Recorded cost: USD 25.527418

Each challenge directory contains:

- `solution_N.py`: candidate produced by the solver;
- `functional-stdout-official.txt`: official functional evaluator output;
- `reward.json`: official verifier scores;
- `audit-details.json`: static, functional, and Codex audit details;
- `job-result.json`: Harbor aggregate result;
- `stamp-info.json`: model, effort, trial, resource, token, and cost metadata.

Harbor's artifact sanitizer replaced the ordinary `true` identifier fragment
in four places in the collected challenge-04 candidate. The archived
`solution_4.py` restores those four unambiguous identifiers
(`true_probabilities`, `true_p01`, and `true_p10`) so the saved candidate
matches the valid source that ran inside the verifier container. Verifier
outputs remain verbatim.

The verifier-side sanitizer also renders some affirmative booleans as the
literal token `[REDACTED]` in `audit-details.json`. Those files are retained
verbatim for provenance but are therefore not strict JSON. Machine-readable
scores are in `reward.json`.
