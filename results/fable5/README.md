# Fable 5 Agent-Axis Benchmark Results

This directory records an agent-axis run of ORBIT-Q with **Fable 5**
(Anthropic, via Cursor agent) as the solver, holding the framework fixed at
**TensorCircuit-NG** (`challenge-benchmark-quantum-tensorcircuit:py311`,
`tensorcircuit-ng 1.7.0.dev`).

## Protocol

- Solver: Fable 5 in a Cursor agent session. For each challenge the solver
  sees only `tasks/challenge-NN/instruction.md` and
  `prompts/frameworks/tensorcircuit.md`. It never reads `tests/` (graders,
  audit) or `solution/` (reference baseline). Prototyping and self-testing run
  inside the framework Docker image against a locally reconstructed config
  derived from the instruction text only.
- Harness note: this is a Cursor-agent run, not a Codex/Claude-Code
  in-container run, so agent-side resource metrics (tokens, solve wall time)
  are not directly comparable with the paper's harness rows. Artifact-side
  metrics (pass components, `runtime_sec`) follow the standard verifier.
- Verification: Harbor verifier-only candidate check (`AGENTS.md`), i.e. the
  unmodified `/tests/test.sh` pipeline: functional evaluator + static policy +
  Codex `gpt-5` LLM audit.

```text
reward = functional_score * static_policy_score * llm_audit_score
```

- Two verification records are kept per challenge:
  - `reward-cloud-precheck.json`: functional + static run on a clean cloud
    Linux VM (4 vCPU, no OpenAI credentials, so `llm_audit_score=0` there by
    construction).
  - `reward.json`: the official full run including the Codex audit, produced
    on the maintainer's machine via `tools/verify_challenge_mac.sh`.

## Files per challenge

```text
challenge-NN/
  solution_N.py                    # Fable 5 candidate artifact
  functional-stdout-cloud.txt      # evaluator output from the cloud precheck
  reward-cloud-precheck.json       # functional/static precheck scores
  functional-stdout-official.txt   # evaluator output from the official stamp
  reward.json                      # official reward incl. Codex audit
```

## Tools

- `tools/verify_challenge_mac.sh NN`: runs the official verifier-only check on
  the maintainer's Mac (colima Docker, local Harbor env, Codex auth via
  `~/.codex/auth.json`, OpenAI reachability through a local proxy relay).
- `tools/proxy_relay.py`: temporary TCP relay `0.0.0.0:7891 ->
  127.0.0.1:7890` so Docker containers can reach the host-only ClashX proxy
  during the audit. Stop it after verification.
