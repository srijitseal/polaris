#!/usr/bin/env bash
# Generate combined XGBoost + CheMeleon comparison figures for all modeling notebooks.
# Usage: bash scripts/combine.sh
#
# Requires both --model xgboost and --model chemeleon runs to have completed first
# (i.e. both predictions.parquet files must exist for each notebook).
# This script does no model training — it only reads existing outputs and produces figures.

set -uo pipefail

trap 'echo ""; echo "  !! Interrupted — aborting remaining jobs"; exit 130' INT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/combine"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

PIXI="pixi run -e cheminformatics python"
NB="notebooks"

# ── Notebook list ──────────────────────────────────────────────────────────
# Each entry: "log_stem|depends_on_stem|script|args..."
# 2.10 sensitivity --combined is not supported; only main has combined figures.
declare -a JOBS=(
    "2.07_combined|-|$NB/2.07-seal-performance-distance.py|--combined"
    "2.08_combined|-|$NB/2.08-seal-baseline-performance.py|--combined"
    "2.09_combined|-|$NB/2.09-seal-iid-vs-ood-series.py|--combined"
    "2.10_combined|-|$NB/2.10-seal-activity-cliff-eval.py|main --combined"
    "2.11_combined|-|$NB/2.11-seal-scaffold-vs-random.py|--combined"
    "2.12_combined|-|$NB/2.12-seal-split-variance.py|--combined"
    "2.13_combined|-|$NB/2.13-seal-molecular-variants.py|--combined"
    "2.15_combined|-|$NB/2.15-zalte-resonance-variants.py|--combined"
)

# ── Helpers ────────────────────────────────────────────────────────────────
PASS=0; FAIL=0; SKIP=0
declare -a FAILED_JOBS=()
declare -A PASSED_STEMS=()

log() { echo "$*" | tee -a "$RUN_LOG"; }

run_job() {
    local stem="$1" depends="$2" script="$3"
    shift 3
    local args=("$@")
    local job_log="$LOG_DIR/${stem}_${TIMESTAMP}.log"

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
log "Combined figures run — $TIMESTAMP"
log "Repo: $REPO_ROOT"
log "Logs: $LOG_DIR"

for job in "${JOBS[@]}"; do
    IFS='|' read -r stem depends script args_str <<< "$job"
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
