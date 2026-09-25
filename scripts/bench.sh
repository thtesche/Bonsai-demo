#!/bin/sh
# Measure this machine's real token/s for the current model, then print a comparison.
#
# The demo's numbers in the docs are from other people's hardware. Decode for a 27B at
# 2 bit is memory-bandwidth-bound, so speed tracks the machine's memory bandwidth far more
# than the flags do. Measure before changing anything: this script exists so that a tuning
# decision is made on numbers from the box it will actually run on.
#
# It measures through the demo's own start scripts, so what it reports is the configuration
# you would get by running them, not a hand-tuned one.
#
# Usage:
#   ./scripts/bench.sh                    # llama.cpp and MLX, 3 runs each
#   ./scripts/bench.sh --backend llama    # one backend only
#   ./scripts/bench.sh --runs 5 --tokens 256
#   ./scripts/bench.sh --port 8099        # if something already owns 8099
#   ./scripts/bench.sh --url http://192.168.0.109:8080   # a server someone else started
#
# --url measures a server that is already running and does not start or stop anything, so it
# is safe against a long-lived instance you did not launch (another machine, or a server in
# production). It is the way to compare two models that are each served on their own port.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"
assert_valid_model
DEMO_DIR="$(resolve_demo_dir)"
cd "$DEMO_DIR"

BACKEND="both"
RUNS=3
MAX_TOKENS=256
PORT=8099
MLX_PORT=8081
URL=""

while [ $# -gt 0 ]; do
    case "$1" in
        --backend) BACKEND="$2"; shift 2 ;;
        --runs)    RUNS="$2"; shift 2 ;;
        --tokens)  MAX_TOKENS="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --url)     URL="$2"; shift 2 ;;
        -h|--help) sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

case "$BACKEND" in
    llama|mlx|both) ;;
    *) err "--backend must be llama, mlx or both."; exit 1 ;;
esac

if [ -n "$URL" ] && [ "$BACKEND" = "mlx" ]; then
    err "--url measures an OpenAI-compatible server, so it cannot be combined with --backend mlx."
    err "  Use --backend llama (the default) to measure $URL."
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    err "python3 is required (used to read the API timings)."
    exit 1
fi

TMP="$(mktemp -d)"
LLAMA_PID=""
TURN2_FOLLOWUP="Now list exactly three practical ways a compiler can improve cache behaviour."

cleanup() {
    if [ -n "$LLAMA_PID" ]; then kill "$LLAMA_PID" 2>/dev/null || true; fi
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

# ── The measured prompt ──
# Long enough that prefill is measurable. A prompt of a few hundred tokens is dominated by
# fixed per-request cost and understates prefill throughput, so the background paragraph is
# repeated to reach a size where the number means something. Fixed, so runs are comparable.
# Each run appends a unique nonce in the client (below) so the prefix is never served from
# the KV cache, which keeps prefill an honest number rather than a cache read.
cat > "$TMP/para.txt" <<'PARA'
Background: modern CPUs execute instructions out of order and speculatively, so a single
load instruction may actually touch several addresses before the machine knows which one
mattered. Without a cache hierarchy every one of those accesses would stall the pipeline on
main memory latency, which on a desktop part is roughly one hundred to two hundred cycles,
several orders of magnitude above the cost of an L1 lookup. Caches exist to convert those
frequent, temporally clustered, spatially clustered accesses into a small number of cheap
hits, and the replacement policy decides what to evict when a level fills up.
PARA

# Conversation padding for the follow-up test, as a multiple of the base prompt.
PAD=8

: > "$TMP/base_prompt.txt"
_i=1
while [ "$_i" -le 3 ]; do
    cat "$TMP/para.txt" >> "$TMP/base_prompt.txt"
    printf '\n' >> "$TMP/base_prompt.txt"
    _i=$((_i + 1))
done
cat >> "$TMP/base_prompt.txt" <<'PROMPT'
Explain in detail how a CPU cache works: define L1, L2 and L3, describe their typical sizes,
latencies and access times on a modern desktop processor, and explain why the hierarchy exists
and how the replacement policies (LRU and friends) work.
PROMPT

# ── Shared client: reports decode and prefill throughput from the server's own timings ──
write_client() {
    cat > "$TMP/client.py" <<'PYEOF'
import json
import os
import secrets
import statistics
import sys
import time
import urllib.request

BASE, MAX_TOKENS, RUNS, PROMPT_FILE = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
MODE = sys.argv[5] if len(sys.argv) > 5 else "single"
PAD = int(sys.argv[6]) if len(sys.argv) > 6 else 8

base = open(PROMPT_FILE).read().strip()
TURN2 = "Now list exactly three practical ways a compiler can improve cache behaviour."


def post(messages, max_tokens):
    body = {
        "model": "local",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
        # off on both backends so the two measure the same work; a thinking model burns
        # most of its time on reasoning tokens and that is a sampling choice, not a
        # hardware measurement
        "thinking_budget_tokens": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.loads(r.read())
    t = res.get("timings", {})
    return {
        "prompt_tokens": res["usage"]["prompt_tokens"],
        "prompt_ms": t.get("prompt_ms", 0),
        "gen_tps": t.get("predicted_per_second", 0),
        "prefill_tps": t.get("prompt_per_second", 0),
        "content": res["choices"][0]["message"]["content"],
    }


def single():
    rows = []
    for _ in range(RUNS):
        prompt = f"[run {secrets.token_hex(4)}]\n{base}"
        r = post([{"role": "user", "content": prompt}], MAX_TOKENS)
        rows.append(r)
        print(
            f"    run: {r['prompt_tokens']:>5} prompt tok | "
            f"prefill {r['prefill_tps']:.1f} t/s | decode {r['gen_tps']:.2f} t/s",
            file=sys.stderr,
        )
    gen = statistics.median(r["gen_tps"] for r in rows)
    pre = statistics.median(r["prefill_tps"] for r in rows)
    print(f"{gen:.2f} {pre:.1f}")


def followup():
    """The turn-2 case: an OpenAI client resends the whole history, and only the
    difference between the backends is who has to read it again.

    The conversation is padded to a few thousand tokens on purpose. A short chat hides the
    effect, because there is little prefix to save; the number that matters is the one at
    the conversation length people actually reach."""
    long_doc = "\n\n".join([base] * PAD)
    prompt = f"[run {secrets.token_hex(4)}]\n{long_doc}"
    t1 = post([{"role": "user", "content": prompt}], 200)
    t2 = post(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": t1["content"]},
            {"role": "user", "content": TURN2},
        ],
        200,
    )
    print(f"{t2['prompt_ms'] / 1000:.2f} {t2['prompt_tokens']}")


try:
    if MODE == "followup":
        followup()
    else:
        single()
except Exception as exc:  # noqa: BLE001 - report and move on, the caller prints the gap
    print(f"ERROR {exc}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

median() {   # median of whitespace-separated numbers on stdin
    tr ' ' '\n' | grep -v '^$' | sort -n \
        | awk '{a[NR]=$1} END{if(NR==0){print "0";exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
}

# ── Measure a server that is already answering on $1, and print the table ──
# Shared by the self-started path and --url, so both produce the same numbers from the
# same client and the same prompt.
measure_url() {
    _mu_base="$1"
    _mu_label="${2:-llama.cpp}"

    write_client
    R=$(python3 "$TMP/client.py" "$_mu_base" "$MAX_TOKENS" "$RUNS" "$TMP/base_prompt.txt" single 2>"$TMP/run.log")
    cat "$TMP/run.log" >&2
    if [ -z "$R" ]; then
        err "no result from $_mu_base; see the run log above"
        return 1
    fi
    R_GEN=$(echo "$R" | cut -d' ' -f1)
    R_PRE=$(echo "$R" | cut -d' ' -f2)
    printf "  %-22s %10s %10s\n" "metric" "decode t/s" "prefill t/s"
    printf "  %-22s %10s %10s\n" "----------------------" "----------" "----------"
    printf "  %-22s %10s %10s\n" "$_mu_label" "$R_GEN" "$R_PRE"

    F=$(python3 "$TMP/client.py" "$_mu_base" "$MAX_TOKENS" 1 "$TMP/base_prompt.txt" followup "$PAD" 2>/dev/null)
    if [ -n "$F" ]; then
        F_SECS=$(echo "$F" | cut -d' ' -f1)
        F_TOK=$(echo "$F" | cut -d' ' -f2)
        printf "  %-22s %10s %10s\n" "follow-up prefill" "${F_SECS}s" "${F_TOK} tok"
    fi
}

# ── An already-running server, started by someone else or on another host ──
bench_url() {
    if ! curl -s --max-time 5 "$URL/health" >/dev/null 2>&1; then
        err "$URL is not answering on /health"
        return 1
    fi
    step "$URL (already running, left untouched)"
    measure_url "$URL" "server"
}

# ── llama.cpp: through the real start script, so the numbers match what you would run ──
bench_llama() {
    if curl -s --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        warn "Something is already serving on port $PORT. Stop it, or pass --port."
        warn "To measure a server you did not start, use --url http://localhost:$PORT"
        return 1
    fi

    BAND="$(select_model_gguf "$GGUF_MODEL_DIR" mac 2>/dev/null | xargs basename 2>/dev/null || echo '?')"
    step "llama.cpp (${BAND:-?})"
    PORT="$PORT" ./scripts/start_llama_server.sh > "$TMP/server.log" 2>&1 &
    LLAMA_PID=$!

    _i=0
    while [ "$_i" -lt 120 ]; do
        if curl -s --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            err "llama-server exited during startup:"
            tail -n 15 "$TMP/server.log" >&2
            LLAMA_PID=""
            return 1
        fi
        _i=$((_i + 1))
        sleep 2
    done
    if ! curl -s --max-time 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then
        err "llama-server did not come up within 240s."
        return 1
    fi
    sleep 5   # let the first request not race the weight load

    measure_url "http://localhost:$PORT" "llama.cpp"

    kill "$LLAMA_PID" 2>/dev/null || true
    LLAMA_PID=""
    sleep 2
}

# ── MLX: the one-shot path, which is the only correct loader for these packs ──
bench_mlx() {
    if [ "$BONSAI_FAMILY" != "bonsai2" ]; then
        warn "MLX path here is verified for Bonsai 2 only; skipping ${BONSAI_DISPLAY}."
        return 1
    fi
    if [ ! -x "$DEMO_DIR/.venv-vlm/bin/python" ]; then
        warn "No .venv-vlm. Create it with ./setup.sh, or:"
        echo "      uv venv .venv-vlm && uv pip install --python .venv-vlm/bin/python \\" >&2
        echo "        -r \"$MLX_MODEL_DIR/runtime/requirements.txt\"" >&2
        return 1
    fi
    if lsof -ti TCP:"$MLX_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        warn "Something is already serving on port $MLX_PORT; the MLX one-shot does not need it."
    fi

    step "MLX (${BONSAI_DISPLAY}-mlx, via the pack's Hadamard-aware loader)"
    PROMPT_TEXT="$(cat "$TMP/base_prompt.txt")"

    GENS=""
    PRES=""
    _i=1
    while [ "$_i" -le "$RUNS" ]; do
        OUT=$(./scripts/run_mlx.sh -p "$PROMPT_TEXT" --no-think -n "$MAX_TOKENS" --stats 2>&1 \
              | sed -e 's/\x1b\[[0-9;]*m//g')
        G=$(echo "$OUT" | sed -n 's/^Generation: [0-9]* tokens @ \([0-9.]*\) t\/s.*/\1/p')
        P=$(echo "$OUT" | sed -n 's/^Prompt: \([0-9]*\) tokens.*/\1/p')
        PT=$(echo "$OUT" | sed -n 's/^Prompt: [0-9]* tokens @ \([0-9.]*\) t\/s.*/\1/p')
        if [ -z "$G" ]; then
            err "MLX run $_i produced no stats (is the pack complete?):"
            echo "$OUT" | tail -n 12 >&2
            return 1
        fi
        echo "    run $_i: prefill $PT t/s ($P tok) | decode $G t/s" >&2
        GENS="$GENS $G"
        PRES="$PRES $PT"
        _i=$((_i + 1))
    done

    MG=$(echo "$GENS" | median | awk '{printf "%.2f", $1}')
    MP=$(echo "$PRES" | median | awk '{printf "%.1f", $1}')

    printf "  %-22s %10s %10s\n" "metric" "decode t/s" "prefill t/s"
    printf "  %-22s %10s %10s\n" "----------------------" "----------" "----------"
    printf "  %-22s %10s %10s\n" "MLX" "$MG" "$MP"

    # mlx-vlm keeps no prompt cache across requests, so a follow-up re-reads the whole
    # conversation. Measure that directly instead of leaving it as a footnote, on a
    # conversation the same size as the llama.cpp side so the two lines are comparable.
    FPROMPT=""
    _i=1
    while [ "$_i" -le "$PAD" ]; do
        if [ -n "$FPROMPT" ]; then FPROMPT="$FPROMPT

"; fi
        FPROMPT="$FPROMPT$PROMPT_TEXT"
        _i=$((_i + 1))
    done
    FPROMPT="$FPROMPT

Assistant: The memory wall is the gap between CPU and main memory speed. Caches exist to close it, by exploiting locality.

$TURN2_FOLLOWUP"
    F=$(./scripts/run_mlx.sh -p "$FPROMPT" --no-think -n 1 --stats 2>&1 | sed -e 's/\x1b\[[0-9;]*m//g')
    FT=$(echo "$F" | sed -n 's/^Prompt: \([0-9]*\) tokens.*/\1/p')
    FP=$(echo "$F" | sed -n 's/^Prompt: [0-9]* tokens @ \([0-9.]*\) t\/s.*/\1/p')
    if [ -n "$FT" ] && [ -n "$FP" ]; then
        F_SECS=$(awk -v t="$FT" -v p="$FP" 'BEGIN{printf "%.1f", t/p}')
        printf "  %-22s %10s %10s\n" "follow-up prefill" "${F_SECS}s" "$FT tok"
    fi
}

echo ""
if [ -n "$URL" ]; then
    echo "=== Token throughput of $URL ==="
else
    echo "=== Token throughput on this machine ==="
fi
if [ -n "$URL" ]; then
    echo "  runs: $RUNS   decode length: $MAX_TOKENS tokens"
else
    echo "  model: ${BONSAI_DISPLAY}   runs: $RUNS   decode length: $MAX_TOKENS tokens"
fi
echo "  thinking is off in both, so this measures the model, not the reasoning budget."
echo ""

case "$BACKEND" in
    llama) if [ -n "$URL" ]; then bench_url; else bench_llama; fi ;;
    mlx)   bench_mlx ;;
    both)  if [ -n "$URL" ]; then bench_url; else bench_llama; echo ""; bench_mlx; fi ;;
esac

echo ""
step "Reading this"
cat <<'NOTES'
  - decode is the number that decides how a long answer feels. For a 27B at 2 bit it is
    limited by how fast the machine can stream weights out of memory, so two Macs with the
    same model can differ by more than any flag will.
  - prefill is what you wait for before the first token. It is usually 3-6x faster than
    decode and rarely the thing that feels slow.
  - the follow-up line is the reason llama-server is the default for chatting. It caches the
    conversation prefix, so turn 2 of a long chat only reads the new text. The MLX one-shot
    re-reads everything every turn, so a long conversation costs a full prefill again for
    every message. If you benchmark only the first turn you will not see this.
  - none of this is a substitute for your own workload. A code or maths question decodes
    differently from prose, and thinking (left off here) can dominate the wall clock.
NOTES
