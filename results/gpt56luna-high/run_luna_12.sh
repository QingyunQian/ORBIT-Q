#!/usr/bin/env bash
# Reproducible one-valid-outcome-per-task GPT-5.6 Luna/high Harbor campaign.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RESULT_ROOT="$ROOT/results/gpt56luna-high"
mkdir -p "$ROOT/jobs"
RUN_ROOT="${LUNA_RUN_ROOT:-$(mktemp -d "$ROOT/jobs/gpt56luna-high-solaudit-20260801.XXXXXX")}"
HARBOR="$ROOT/.conda/harbor-py312/bin/harbor"
IMAGE="challenge-benchmark-quantum-tensorcircuit:py311"
AUDIT_CONFIG="$RESULT_ROOT/audit-high.config.toml"
BASE_COMMIT="${BENCHMARK_BASE_COMMIT:-0201238ec2983907e2891f5319f5fff2d00844d5}"
PROXY="${BENCHMARK_PROXY:-http://172.17.0.1:7892}"

if (($#)); then
  challenges=("$@")
else
  challenges=(01 02 03 04 05 06 07 08 09 10 11 12)
fi

if [[ ! -f "$RUN_ROOT/task-copy-manifest.json" ]]; then
  python3 "$RESULT_ROOT/tools/prepare_tasks.py" \
    --repo-root "$ROOT" \
    --run-root "$RUN_ROOT" \
    --source-run-root "$ROOT/jobs/gpt56terra-high-solaudit-20260731-valid" \
    --base-commit "$BASE_COMMIT"
fi

printf 'Run root: %s\n' "$RUN_ROOT"
printf 'Base commit: %s\n' "$BASE_COMMIT"
printf 'Solver: gpt-5.6-luna/high; audit: gpt-5.6-sol/high\n'

overall_rc=0
for raw in "${challenges[@]}"; do
  nn="$(printf '%02d' "$((10#$raw))")"
  task="$RUN_ROOT/challenge-$nn"
  valid=0
  attempt=1
  while [[ -d "$RUN_ROOT/jobs/challenge-$nn-tensorcircuit-gpt-5.6-luna-high-20260801-r$attempt" ]]; do
    attempt=$((attempt + 1))
  done
  for retry in 1 2 3; do
    job="challenge-$nn-tensorcircuit-gpt-5.6-luna-high-20260801-r$attempt"
    log="$RUN_ROOT/host-challenge-$nn-r$attempt.log"

    printf '\n[%s] START challenge-%s attempt=%s\n' "$(date -u +%FT%TZ)" "$nn" "$attempt"
    if PYTHONPATH="$ROOT" "$HARBOR" run \
      -p "$task" \
      --extra-instruction-path "$ROOT/prompts/frameworks/tensorcircuit.md" \
      --environment-import-path adapters.framework_docker:FrameworkDockerEnvironment \
      --environment-kwarg framework=tensorcircuit \
      --environment-kwarg "docker_image=$IMAGE" \
      --agent-import-path harbor.agents.installed.codex:Codex \
      --agent-kwarg reasoning_effort=high \
      --agent-env CODEX_FORCE_AUTH_JSON=true \
      --agent-env "HTTP_PROXY=$PROXY" \
      --agent-env "HTTPS_PROXY=$PROXY" \
      --agent-env NO_PROXY=localhost,127.0.0.1 \
      --verifier-import-path adapters.codex_para_verifier:CodexParaVerifier \
      --verifier-kwarg audit_model=gpt-5.6-sol \
      --verifier-kwarg force_auth_json=true \
      --verifier-kwarg "profile_config_path=$AUDIT_CONFIG" \
      --verifier-env REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit \
      --verifier-env "HTTP_PROXY=$PROXY" \
      --verifier-env "HTTPS_PROXY=$PROXY" \
      --verifier-env NO_PROXY=localhost,127.0.0.1 \
      -m gpt-5.6-luna \
      -n 1 \
      -o "$RUN_ROOT/jobs" \
      --job-name "$job" \
      --yes 2>&1 | tee "$log"
    then
      rc=0
    else
      rc=$?
    fi
    printf '[%s] END challenge-%s attempt=%s rc=%s\n' \
      "$(date -u +%FT%TZ)" "$nn" "$attempt" "$rc"

    reward="$(find "$RUN_ROOT/jobs/$job" -path '*/verifier/reward.json' -type f -print -quit 2>/dev/null)"
    agent_log="$(find "$RUN_ROOT/jobs/$job" -path '*/agent/codex.txt' -type f -print -quit 2>/dev/null)"
    infra_failure=0
    if [[ -n "$agent_log" ]] \
      && rg -q '"type":"turn.failed"' "$agent_log" \
      && rg -q \
        'tls handshake eof|stream disconnected before completion|error sending request for url|failed to connect to websocket|connection reset by peer|temporary failure in name resolution' \
        "$agent_log"; then
      infra_failure=1
      printf '[%s] INFRA challenge-%s attempt=%s: solver transport failure\n' \
        "$(date -u +%FT%TZ)" "$nn" "$attempt"
    fi
    if [[ -n "$reward" && -s "$reward" && "$infra_failure" -eq 0 ]]; then
      valid=1
      break
    fi
    printf '[%s] INVALID challenge-%s attempt=%s: valid outcome missing; retrying\n' \
      "$(date -u +%FT%TZ)" "$nn" "$attempt"
    attempt=$((attempt + 1))
  done

  if ((valid == 0)); then
    printf '[%s] ABORT challenge-%s: no valid outcome after 3 attempts\n' \
      "$(date -u +%FT%TZ)" "$nn"
    overall_rc=1
    break
  fi
done

exit "$overall_rc"
