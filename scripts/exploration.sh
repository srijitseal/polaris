#!/usr/bin/env bash
# Run all exploration/preprocessing notebooks sequentially, logging each to logs/exploration/.
# These notebooks do not train models — they produce interim data, split files, and
# characterization figures. Run this before any model training script.
#
# Usage: bash scripts/exploration.sh
#
# Order matters: 1.01 (data loader) must run before split notebooks (2.03–2.06),
# and 2.14 (resonance generation) must run before 2.15 model training.

set -uo pipefail

trap 'echo ""; echo "  !! Interrupted — aborting remaining jobs"; exit 130' INT

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/exploration"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/run_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

PIXI="pixi run -e cheminformatics python"
NB="notebooks"

# ── Notebook list ──────────────────────────────────────────────────────────
# Each entry: "log_stem|depends_on_stem|script|args..."
# 1.01 must run first (produces interim parquet files everything else reads).
# 2.14 (resonance generation) must run before 2.15 model training.
declare -a JOBS=(
    # ── Data loading (prerequisite for everything) ──────────────────────
    "1.01_dataset_loader|-|$NB/1.01-seal-dataset-loader.py|"

    # ── Initial explorations ────────────────────────────────────────────
    "0.01_dataset_exploration|1.01_dataset_loader|$NB/0.01-seal-dataset-exploration.py|"
    "0.02_ecfp_distance|1.01_dataset_loader|$NB/0.02-seal-ecfp-distance-exploration.py|"
    "0.03_chembl_tanimoto|-|$NB/0.03-araripe-chembl-tanimoto.py|"

    # ── Dataset characterization ────────────────────────────────────────
    # 2.01 reads tanimoto_distance_matrix.npz → depends on 0.02
    "2.01_chemical_space|0.02_ecfp_distance|$NB/2.01-seal-chemical-space-analysis.py|"
    # 2.02 reads only expansion_tx.parquet → depends on 1.01
    "2.02_target_distribution|1.01_dataset_loader|$NB/2.02-seal-target-distribution.py|"

    # ── Split generation (produces fold parquets used by model notebooks) ─
    # 2.03/2.04/2.05 read tanimoto_distance_matrix.npz → depend on 0.02
    "2.03_cluster_split|0.02_ecfp_distance|$NB/2.03-seal-cluster-split.py|"
    "2.04_time_split|0.02_ecfp_distance|$NB/2.04-seal-time-split.py|"
    "2.05_target_split|0.02_ecfp_distance|$NB/2.05-seal-target-split.py|"

    # ── Split quality analysis ──────────────────────────────────────────
    "2.06_split_quality|2.03_cluster_split|$NB/2.06-seal-split-quality.py|"

    # ── Resonance form generation (prerequisite for 2.15 model training) ─
    "2.14_resonance_generation|1.01_dataset_loader|$NB/2.14-zalte-resonance-generation.py|"

    # ── Additional analyses ─────────────────────────────────────────────
    # 2.16 reads expansion_tx.parquet (1.01) + tanimoto_distance_matrix.npz (0.02)
    "2.16_scaffold_boundary|0.02_ecfp_distance|$NB/2.16-araripe-scaffold-boundary.py|"
    "3.02_framework_figure|-|$NB/3.02-seal-framework-figure.py|"
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
    log "  cmd:   $PIXI $script${args[*]:+ ${args[*]}}"
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
log "Exploration run — $TIMESTAMP"
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
