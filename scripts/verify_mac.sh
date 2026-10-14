#!/usr/bin/env bash
# One-command verification for macOS (also works on Linux).
#
#   ./scripts/verify_mac.sh             # Docker checks + real Groq eval (asks for your key)
#   ./scripts/verify_mac.sh --skip-eval # Docker checks only
#   ./scripts/verify_mac.sh --skip-stack
#
# What it does:
#   1. Builds the Docker image
#   2. Smoke-tests the container: health, chat, streaming, auth, metrics
#   3. Starts the full stack (gateway + Prometheus + Grafana) and checks the scrape and dashboard
#   4. Runs the routing eval against real Groq models and writes the results into README.md
#   5. Writes docs/VERIFICATION.md with every check's outcome
#
# Only Docker Desktop is required. Python, the eval and the README update all run inside the container.
# Compatible with macOS's default bash 3.2.

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT="$(pwd)"

SKIP_EVAL=0
SKIP_STACK=0
for arg in "$@"; do
  case "$arg" in
    --skip-eval) SKIP_EVAL=1 ;;
    --skip-stack) SKIP_STACK=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done

IMAGE="llm-gateway:local"
ADMIN_KEY="sk-admin-verify-$$"
NET="llmgw-verify-net"
SMOKE="llmgw-smoke"
GROQ_GW="llmgw-groq"
RESULTS=()
FAILED=0

green() { printf '\033[32m%s\033[0m\n' "$*"; }
red()   { printf '\033[31m%s\033[0m\n' "$*"; }
bold()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
pass()  { green "  PASS  $1"; RESULTS+=("| PASS | $1 |"); }
fail()  { red   "  FAIL  $1"; RESULTS+=("| FAIL | $1 |"); FAILED=$((FAILED + 1)); }
skip()  { echo  "  SKIP  $1"; RESULTS+=("| SKIP | $1 |"); }
check() { local name="$1"; shift; if "$@" >/dev/null 2>&1; then pass "$name"; else fail "$name"; fi; }

cleanup() {
  docker rm -f "$SMOKE" "$GROQ_GW" >/dev/null 2>&1
  docker network rm "$NET" >/dev/null 2>&1
}
trap cleanup EXIT

wait_for() {  # wait_for <url> <seconds>
  local i=0
  while [ "$i" -lt "$2" ]; do
    curl -sf "$1" >/dev/null 2>&1 && return 0
    sleep 1; i=$((i + 1))
  done
  return 1
}

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

# ---------------------------------------------------------------- 0. prereqs
bold "0. Prerequisites"
if ! command -v docker >/dev/null 2>&1; then
  red "Docker is not installed. Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  red "Docker is installed but not running. Open Docker Desktop, wait for 'Engine running', then rerun."
  exit 1
fi
DOCKER_VERSION="$(docker version --format '{{.Server.Version}}' 2>/dev/null)"
pass "Docker engine running (v$DOCKER_VERSION)"

# ------------------------------------------------------------------ 1. build
bold "1. Build image"
if docker build -t "$IMAGE" . ; then
  pass "docker build"
else
  fail "docker build"
  red "Build failed; the remaining checks need the image. See the output above."
  exit 1
fi

# ------------------------------------------------------------------ 2. smoke
bold "2. Container smoke test"
docker rm -f "$SMOKE" >/dev/null 2>&1
docker run -d --name "$SMOKE" -p 127.0.0.1::8000 -e GATEWAY_ADMIN_KEY="$ADMIN_KEY" "$IMAGE" >/dev/null
PORT="$(docker port "$SMOKE" 8000/tcp | head -1 | sed 's/.*://')"
BASE="http://127.0.0.1:$PORT"

if wait_for "$BASE/health" 30; then
  pass "container starts and /health responds"

  chat() {  # chat <api-key> <json-body> [extra curl args...] -> headers+body on stdout
    local key="$1" body="$2"; shift 2
    curl -s -i "$BASE/v1/chat/completions" -H "Authorization: Bearer $key" \
      -H "Content-Type: application/json" "$@" -d "$body" | tr -d '\r'
  }
  has() { printf '%s\n' "$1" | grep -qi -- "$2"; }

  OUT="$(chat sk-demo-dev '{"model":"auto","messages":[{"role":"user","content":"What is the capital of France?"}]}')"
  t_chat()    { has "$OUT" '^HTTP/1.1 200' && has "$OUT" '"object":"chat.completion"'; }
  t_headers() { has "$OUT" '^x-gateway-model:' && has "$OUT" '^x-route-reason:'; }
  check "chat completion returns 200 + OpenAI-shaped body" t_chat
  check "routing headers present (X-Gateway-Model, X-Route-Reason)" t_headers

  OUT2="$(chat sk-demo-dev '{"model":"auto","messages":[{"role":"user","content":"what is the capital of france"}]}')"
  t_cache() { has "$OUT2" '^x-cache: HIT'; }
  check "repeat question served from cache (X-Cache: HIT)" t_cache

  STREAM="$(curl -sN "$BASE/v1/chat/completions" -H "Authorization: Bearer sk-demo-dev" \
    -H "Content-Type: application/json" -d '{"stream":true,"messages":[{"role":"user","content":"stream test"}]}')"
  t_stream() { has "$STREAM" 'data: \[DONE\]' && has "$STREAM" 'chat.completion.chunk'; }
  check "streaming (SSE) chunks end with [DONE]" t_stream

  CODE="$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/chat/completions" \
    -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"hi"}]}')"
  check "missing API key is rejected (401)" test "$CODE" = "401"

  CHAOS="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/admin/providers/mock-alpha/chaos" \
    -H "Authorization: Bearer $ADMIN_KEY" -H "Content-Type: application/json" -d '{"failure_rate":1.0}')"
  OUT3="$(chat sk-demo-dev '{"model":"alpha-mini","messages":[{"role":"user","content":"fallback test"}]}' \
    -H "X-Cache-Bypass: 1")"
  t_fallback() { [ "$CHAOS" = 200 ] && has "$OUT3" '^HTTP/1.1 200' && has "$OUT3" '^x-gateway-model: beta-mini'; }
  check "fallback: provider forced down -> request still succeeds on another model" t_fallback

  t_metrics() { curl -s "$BASE/metrics" | grep -q '^gateway_requests_total'; }
  check "/metrics exposes gateway_requests_total" t_metrics
  t_nonroot() { [ "$(docker exec "$SMOKE" id -u)" != "0" ]; }
  check "container runs as non-root user" t_nonroot
  t_healthy() {
    local i
    for i in $(seq 1 45); do
      [ "$(docker inspect -f '{{.State.Health.Status}}' "$SMOKE")" = healthy ] && return 0
      sleep 1
    done
    return 1
  }
  echo "  (waiting for Docker HEALTHCHECK, up to 45s)"
  check "Docker HEALTHCHECK reports healthy" t_healthy
else
  fail "container starts and /health responds"
  docker logs "$SMOKE" 2>&1 | tail -20
fi
docker rm -f "$SMOKE" >/dev/null 2>&1

# ------------------------------------------------------------------ 3. stack
bold "3. Full stack (gateway + Prometheus + Grafana)"
if [ "$SKIP_STACK" = 1 ]; then
  skip "full stack (--skip-stack)"
elif port_busy 8000 || port_busy 9090 || port_busy 3000; then
  skip "full stack: port 8000, 9090 or 3000 is already in use on this Mac (stop that app and rerun)"
else
  if docker compose up -d --build; then
    pass "docker compose up"
    if wait_for "http://localhost:8000/health" 60; then
      pass "gateway reachable through compose"
      for i in $(seq 1 30); do
        curl -s localhost:8000/v1/chat/completions -H "Authorization: Bearer sk-admin-dev" \
          -H "Content-Type: application/json" \
          -d "{\"messages\":[{\"role\":\"user\",\"content\":\"stack traffic $i\"}]}" >/dev/null
      done
      poll() {  # poll <seconds> <url> <pattern>
        local i
        for i in $(seq 1 "$1"); do
          curl -s "$2" | grep -q -- "$3" && return 0
          sleep 1
        done
        return 1
      }
      echo "  (waiting for Prometheus to scrape and Grafana to start, up to ~2 min)"
      check "Prometheus is scraping the gateway (target up)" \
        poll 60 "http://localhost:9090/api/v1/targets" '"health":"up"'
      check "Prometheus has gateway metrics data" \
        poll 30 "http://localhost:9090/api/v1/query?query=sum(gateway_requests_total)" '"value"'
      check "Grafana is up with the 'LLM Gateway' dashboard provisioned" \
        poll 90 "http://localhost:3000/api/search?query=LLM" 'llm-gateway'
    else
      fail "gateway reachable through compose"
      docker compose logs gateway | tail -20
    fi
    docker compose down >/dev/null 2>&1
  else
    fail "docker compose up"
  fi
fi

# ------------------------------------------------------------------- 4. eval
bold "4. Real routing eval on Groq"
EVAL_DONE=0
if [ "$SKIP_EVAL" = 1 ]; then
  skip "Groq eval (--skip-eval)"
else
  if [ -z "${GROQ_API_KEY:-}" ] && [ -f .env ]; then
    GROQ_API_KEY="$(grep -E '^GROQ_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
  fi
  if [ -z "${GROQ_API_KEY:-}" ]; then
    echo "  Paste your Groq API key (starts with gsk_; input is hidden; press Enter to skip)."
    echo "  Get a free key at https://console.groq.com/keys"
    printf "  GROQ_API_KEY: "
    read -rs GROQ_API_KEY
    echo
  fi
  export GROQ_API_KEY

  if [ -z "$GROQ_API_KEY" ]; then
    skip "Groq eval (no API key given)"
  else
    docker network create "$NET" >/dev/null 2>&1
    docker run -d --name "$GROQ_GW" --network "$NET" -e GATEWAY_CONFIG=config.groq.yaml \
      -e GATEWAY_ADMIN_KEY="$ADMIN_KEY" -e GROQ_API_KEY \
      -e GROQ_BASE_URL="${GROQ_BASE_URL:-https://api.groq.com/openai/v1}" "$IMAGE" >/dev/null
    sleep 3
    # one real request first, to fail fast on a bad key / network issue
    PROBE="$(docker run --rm --network "$NET" "$IMAGE" python -c "
import httpx, sys
r = httpx.post('http://$GROQ_GW:8000/v1/chat/completions', timeout=60,
    headers={'Authorization': 'Bearer $ADMIN_KEY'},
    json={'model': 'llama-8b', 'messages': [{'role': 'user', 'content': 'Say OK'}]})
print(r.status_code, r.text[:300])
" 2>&1)"
    if echo "$PROBE" | grep -q '^200'; then
      pass "Groq key works (real completion from llama-3.1-8b-instant)"
      echo "  Running 60 real requests, paced for Groq's free tier (~3-4 minutes)..."
      if docker run --rm --network "$NET" --user root -v "$ROOT":/work -w /work "$IMAGE" \
          python evals/run_eval.py --url "http://$GROQ_GW:8000" --key "$ADMIN_KEY" \
          --config config.groq.yaml --models auto,llama-70b --rpm 20 --concurrency 3 \
          --out docs/eval_groq.json; then
        pass "eval completed (docs/eval_groq.json)"
        if docker run --rm --user root -v "$ROOT":/work -w /work "$IMAGE" \
            python scripts/update_readme_results.py docs/eval_groq.json; then
          pass "README results section updated with real numbers"
          EVAL_DONE=1
        else
          fail "README update"
        fi
      else
        fail "eval run"
      fi
    else
      fail "Groq key check: $(echo "$PROBE" | head -c 200)"
      echo "  Check the key at https://console.groq.com/keys (and that you're online)."
    fi
    docker rm -f "$GROQ_GW" >/dev/null 2>&1
  fi
fi

# ----------------------------------------------------------------- 5. report
bold "5. Report"
mkdir -p docs
{
  echo "# Verification report"
  echo
  echo "Generated by \`scripts/verify_mac.sh\` on $(date '+%Y-%m-%d %H:%M %Z')."
  echo
  echo "- Machine: $(uname -sm), Docker $DOCKER_VERSION"
  echo "- Image: \`$IMAGE\`"
  echo
  echo "| Result | Check |"
  echo "|---|---|"
  for line in "${RESULTS[@]}"; do echo "$line"; done
  if [ "$EVAL_DONE" = 1 ] && [ -f docs/resume_bullet.txt ]; then
    echo
    echo "Suggested resume bullet from the real eval:"
    echo
    echo "> $(cat docs/resume_bullet.txt)"
  fi
} > docs/VERIFICATION.md
green "Wrote docs/VERIFICATION.md"

if [ "$FAILED" -eq 0 ]; then
  green "All checks passed."
  echo "Next: review the README diff (git diff README.md), then commit:"
  echo "  git add -A && git commit -m 'Verify Docker stack and add real Groq eval results'"
else
  red "$FAILED check(s) failed. See above, or send docs/VERIFICATION.md to Claude."
  exit 1
fi
