#!/bin/bash

set -e

BASE_DIR="/home/tommy/Project/PcodeBERT"
cd "$BASE_DIR"

LOSSES=("mse" "cosine")
LAYERS=(6)

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
log "Adapter Training: 30 epochs (checkpoints at 10, 20, 30)"
log "Losses: ${LOSSES[@]}"
log "Layers: ${LAYERS[@]}"
log "Log: $LOG_FILE"
log ""

log "======================================"
log "Step 1: Training Adapters"
log "======================================"

python src/scripts/6_adapter_ablation.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "✓ All adapter training completed"
else
    log "✗ Adapter training failed"
    exit 1
fi

log ""
log "======================================"
log "Step 2: Applying Adapters"
log "======================================"

for epoch in 10 20 30; do
    for loss in "${LOSSES[@]}"; do
        for layer in "${LAYERS[@]}"; do
            log ""
            log "Applying: epoch=$epoch, loss=$loss, layers=$layer"
            
            python src/scripts/7_apply_adapter_ablation.py \
                --epoch $epoch \
                --loss $loss \
                --layers $layer 2>&1 | tee -a "$LOG_FILE"
            
            if [ $? -eq 0 ]; then
                log "✓ Completed embedding generation"
            else
                log "✗ Embedding generation failed"
                exit 1
            fi
        done
    done
done

log ""
log "======================================"
log "Step 3: Training GNN Models"
log "======================================"

for epoch in 10 20 30; do
    for loss in "${LOSSES[@]}"; do
        for layer in "${LAYERS[@]}"; do
            log ""
            log "GNN training: epoch=$epoch, loss=$loss, layers=$layer"
            
            python src/scripts/5_gnn_adapter_ablation.py \
                --epoch $epoch \
                --loss $loss \
                --layers $layer 2>&1 | tee -a "$LOG_FILE"
            
            if [ $? -eq 0 ]; then
                log "✓ Completed GNN training"
            else
                log "✗ GNN training failed"
                exit 1
            fi
        done
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
log "Total configurations: 6 (3 epochs × 2 losses × 1 layer)"
log "Results: outputs/results/adapter_ablation/"
log "Log: $LOG_FILE"
log ""
