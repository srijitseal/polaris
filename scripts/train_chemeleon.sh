#!/usr/bin/env bash
# Run all CheMeleon training notebooks sequentially, logging each to logs/chemeleon/.
# Usage: bash scripts/train_chemeleon.sh
# Skips any notebook whose predictions.parquet already exists.

set -uo pipefail

# Ctrl+C aborts the whole run, not just the current job.
_interrupted=0
trap '_interrupted=1; echo ""; echo "  !! Interrupted — aborting remaining jobs"; exit 130' INT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/chemeleon"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

PIXI="pixi run -e cheminformatics python"
NB="notebooks"

# ── Notebook list ──────────────────────────────────────────────────────────
# Each entry: "log_stem|depends_on_stem|script|args..."
# depends_on_stem: name of a prior stem that must have PASSED, or "-" for none.
# 2.10 sensitivity depends on 2.10 main completing successfully.
declare -a JOBS=(
    "2.07_chemeleon|-|$NB/2.07-seal-performance-distance.py|--model chemeleon"
    "2.08_chemeleon|-|$NB/2.08-seal-baseline-performance.py|--model chemeleon"
    "2.09_chemeleon|-|$NB/2.09-seal-iid-vs-ood-series.py|--model chemeleon"
    "2.10_chemeleon_main|-|$NB/2.10-seal-activity-cliff-eval.py|main --model chemeleon"
    "2.10_chemeleon_sensitivity|2.10_chemeleon_main|$NB/2.10-seal-activity-cliff-eval.py|sensitivity --model chemeleon"
    "2.11_chemeleon|-|$NB/2.11-seal-scaffold-vs-random.py|--model chemeleon"
    "2.12_chemeleon|-|$NB/2.12-seal-split-variance.py|--model chemeleon"
    "2.13_chemeleon|-|$NB/2.13-seal-molecular-variants.py|--model chemeleon"
    "2.15_chemeleon|-|$NB/2.15-zalte-resonance-variants.py|--model chemeleon"
)

# ── Helpers ────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
declare -a FAILED_JOBS=()
declare -A PASSED_STEMS=()  # tracks which stems completed successfully

log() { echo "$*" | tee -a "$RUN_LOG"; }

run_job() {
    local stem="$1" depends="$2" script="$3"
    shift 3
    local args=("$@")
    local job_log="$LOG_DIR/${stem}_${TIMESTAMP}.log"

    # Skip if a required predecessor failed or was skipped.
    if [[ "$depends" != "-" ]] && [[ -z "${PASSED_STEMS[$depends]+x}" ]]; then
        log ""
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "  SKIP   $stem  (dependency '$depends' did not pass)"
        (( SKIP++ )) || true
        FAILED_JOBS+=("$stem (skipped: $depends failed)")
        return
    fi

    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  START  $stem"
    log "  cmd:   $PIXI $script ${args[*]}"
    log "  log:   $job_log"
    log "  time:  $(date)"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $PIXI "$script" "${args[@]}" 2>&1 | tee "$job_log"; then
        log "  ✓ DONE  $stem  ($(date))"
        (( PASS++ )) || true
        PASSED_STEMS[$stem]=1
    else
        local rc=$?
        log "  ✗ FAIL  $stem  (exit $rc)"
        (( FAIL++ )) || true
        FAILED_JOBS+=("$stem")
    fi
}

# ── Main ───────────────────────────────────────────────────────────────────
log "CheMeleon training run — $TIMESTAMP"
log "Repo: $REPO_ROOT"
log "Logs: $LOG_DIR"

for job in "${JOBS[@]}"; do
    IFS='|' read -r stem depends script args_str <<< "$job"
    # Split args string into array
    read -ra args <<< "$args_str"
    run_job "$stem" "$depends" "$script" "${args[@]}"
done

# ── Summary ────────────────────────────────────────────────────────────────
log ""
log "══════════════════════════════════════════════════════════════"
log "  SUMMARY  $TIMESTAMP"
log "  passed: $PASS   failed: $FAIL"
if [[ ${#FAILED_JOBS[@]} -gt 0 ]]; then
    log "  FAILED JOBS:"
    for j in "${FAILED_JOBS[@]}"; do
        log "    - $j"
    done
fi
log "══════════════════════════════════════════════════════════════"

[[ $FAIL -eq 0 ]]
