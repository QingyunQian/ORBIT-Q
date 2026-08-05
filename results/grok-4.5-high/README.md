# Grok 4.5 High — TensorCircuit Benchmark

Grok 4.5 at `high` solver effort produced **9 valid solutions out of 12**. GPT-5.6 Sol at `high` independently audited every completed candidate.

![Final task outcomes](figs/grok-4.5-high-outcomes.png)

## Protocol

- Solver: `grok-4.5`, `reasoning_effort=high`, through Codex CLI and the xAI API
- Auditor: `gpt-5.6-sol`, audit reasoning `high`
- Framework: TensorCircuit-NG in `challenge-benchmark-quantum-tensorcircuit:py311`
- Frozen base commit: `0201238ec2983907e2891f5319f5fff2d00844d5`
- Frozen task-copy SHA-256: `19fe27b83eaf668b3df32d1a68902b08cbe28585f189290769018eb16d927895`
- Resources: 6 CPU, 10,240 MiB memory, 16,384 MiB storage
- Tasks executed sequentially, with one selected non-infrastructure outcome per task
- Agent limit: 1,800 seconds; submitted solutions still proceed to verification

The local xAI compatibility layer only repairs integer declarations in tool JSON schemas and converts mathematically integral tool arguments to JSON integers where the active schema requires an integer. It does not alter prompts, semantic values, tool outputs, candidates, or task data.

## Results

| Task | Attempt | Outcome | Functional | Static | Audit | Runtime (s) |
|---:|:---:|:---:|---:|---:|---:|---:|
| 01 | r1 | Pass | 1 | 1 | 1 | 160.68 |
| 02 | r1 | Pass | 1 | 1 | 1 | 57.09 |
| 03 | r1 | Pass | 1 | 1 | 1 | 5.11 |
| 04 | r1 | Fail | 1 | 0 | 0 | 34.26 |
| 05 | r1 | Pass | 1 | 1 | 1 | 180.34 |
| 06 | r2 | Pass | 1 | 1 | 1 | 38.08 |
| 07 | r3 | Pass | 1 | 1 | 1 | 121.61 |
| 08 | r1 | Fail | 0 | 0 | 0 | — |
| 09 | r1 | Pass | 1 | 1 | 1 | 77.90 |
| 10 | r1 | Pass | 1 | 1 | 1 | 89.31 |
| 11 | r1 | Fail | 0 | 0 | 0 | — |
| 12 | r1 | Pass | 1 | 1 | 1 | 11.13 |

Ten tasks produced measured candidate runtimes totaling 775.51 seconds. Runtime is reported separately and does not reduce pass reward.

### Failed tasks

- **Task 04:** the program was functionally correct, but its 281-line implementation put the core noisy-circuit evolution and gradients in a custom NumPy Pauli-transfer-matrix simulator. Static policy and Sol/high audit rejected this TensorCircuit bypass.
- **Task 08:** while recovering from hung profiling commands, Grok issued `pkill -f 'python'`. The broad full-command-line match terminated its own agent wrapper/proxy chain (exit 143).
- **Task 11:** Grok similarly issued `pkill -f "jax"` while trying to stop a long compilation, again terminating its own agent chain (exit 143).

Tasks 08 and 11 are retained as valid model tool-use failures. They were not infrastructure failures and were not rerun.

## Agent-side resources

![Grok 4.5 high agent-side resource use](figs/grok-4.5-high-agent-resource-use.png)

The 12 selected outcomes used 187.9 minutes of Agent wall time and 39.911 million solving-side tokens: 1.211 million non-cache-read prompt tokens, 38.384 million cache-read prompt tokens, and 0.316 million output tokens. This is 20.88 Agent minutes and 4.434 million tokens per valid solution. The integration did not report xAI provider cost, so the comparison uses time and tokens rather than fabricating a dollar estimate.

## Provenance

Each `challenge-NN/` directory contains the selected solution when one exists, reward and audit outputs, functional output, complete solver log and session stream, xAI proxy log, Harbor configs/results, hashes, and a normalized `stamp-info.json`. Tasks 08 and 11 additionally contain `model-failure.json`.

Excluded attempts are documented in `summary.json`: Task 06 r1 had the pre-repair integer-argument compatibility failure, while Task 07 r1/r2 were interrupted during that diagnosis. No task with a selected valid outcome was rerun.
