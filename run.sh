#!/bin/bash

set -e

BASE_DIR="/home/tommy/Project/PcodeBERT"
cd "$BASE_DIR"

LOSSES=("cosine" "mse")
EPOCHS=(1 2 3 4 5 6 7 8)

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/adapter_ablation_${TIMESTAMP}.log"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================"
log "Adapter Ablation Experiments"
log "======================================"
log "Pretrain Model: epoch 25"
log "Adapter Training: 10 epochs (saving checkpoints at epochs 1-10)"
log "Losses: ${LOSSES[@]}"
log "Total Configurations: 20 (10 epochs × 2 losses)"
log "Random Seeds: 5 (42, 123, 2025, 31415, 8888)"
log "Log: $LOG_FILE"
log ""
log ""
log "======================================"
log "Step 3: Training GNN Models"
log "======================================"

for epoch in "${EPOCHS[@]}"; do
    for loss in "${LOSSES[@]}"; do
        log ""
        log "GNN training: epoch=$epoch, loss=$loss"
        
        python src/scripts/5_gnn_adapter_ablation.py \
            --epoch $epoch \
            --loss $loss 2>&1 | tee -a "$LOG_FILE"
        
        if [ $? -eq 0 ]; then
            log "✓ Completed GNN training for ${loss}_epoch${epoch}"
        else
            log "✗ GNN training failed for ${loss}_epoch${epoch}"
            exit 1
        fi
    done
done

log ""
log "======================================"
log "Step 4: Aggregating Results"
log "======================================"

python src/scripts/aggregate_ablation_results.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "✓ Results aggregated successfully"
else
    log "✗ Results aggregation failed"
fi

log ""
log "======================================"
log "Adapter Ablation Completed!"
log "======================================"
log "Total configurations: 20 (10 epochs × 2 losses)"
log "Each configuration tested with 5 random seeds"
log "Results: outputs/results/"
log "Aggregated results: outputs/results/ablation_summary/"
log "Log: $LOG_FILE"
log ""