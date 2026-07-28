#!/usr/bin/env bash
set -uo pipefail

ROOT="/Users/qqy/Desktop/2026Project/ORBIT-Q"
RUN_ROOT="/private/tmp/orbitq-gpt56sol-ultra-clean-20260728.4vqkrt"
AUDIT_CONFIG="$RUN_ROOT/audit-high.config.toml"
HARBOR="$ROOT/.conda/harbor-py312/bin/harbor"
IMAGE="challenge-benchmark-quantum-tensorcircuit:py311"
PROXY="http://192.168.5.2:7891"

if (($#)); then
  challenges=("$@")
else
  challenges=(01 02 03 04 05 06 07 08 09 10 11 12)
fi

overall_rc=0
for raw in "${challenges[@]}"; do
  nn="$(printf '%02d' "$((10#$raw))")"
  task="$RUN_ROOT/challenge-$nn"
  job="challenge-$nn-tensorcircuit-gpt-5.6-sol-ultra-clean-20260728-r1"
  log="$RUN_ROOT/host-challenge-$nn.log"

  printf '\n[%s] START challenge-%s\n' "$(date -u +%FT%TZ)" "$nn"
  if PYTHONPATH="$ROOT" "$HARBOR" run \
    -p "$task" \
    --extra-instruction-path "$ROOT/prompts/frameworks/tensorcircuit.md" \
    --environment-import-path adapters.framework_docker:FrameworkDockerEnvironment \
    --environment-kwarg framework=tensorcircuit \
    --environment-kwarg "docker_image=$IMAGE" \
    --agent-import-path harbor.agents.installed.codex:Codex \
    --agent-kwarg reasoning_effort=ultra \
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
    -m gpt-5.6-sol \
    -n 1 \
    -o "$ROOT/jobs" \
    --job-name "$job" \
    --yes 2>&1 | tee "$log"
  then
    rc=0
  else
    rc=$?
    overall_rc=$rc
  fi
  printf '[%s] END challenge-%s rc=%s\n' "$(date -u +%FT%TZ)" "$nn" "$rc"
done

exit "$overall_rc"
