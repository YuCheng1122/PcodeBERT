#!/bin/bash

# Ablation Experiments Automation Script
# This script runs embedding generation and GNN training for different pretrain epochs

set -e  # Exit on error

BASE_DIR="/home/tommy/Project/PcodeBERT"
cd "$BASE_DIR"

# Define pretrain epochs to test
EPOCHS=(25 50 100)

# Source and target CPUs
SOURCE_CPUS="x86_64"
TARGET_CPUS="ARM PPC MIPS x86_64"

# Create log directory and file
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/ablation_experiment_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "======================================"
log "Starting Ablation Experiments"
log "======================================"
log "Pretrain Epochs: ${EPOCHS[@]}"
log "Source CPU: $SOURCE_CPUS"
log "Target CPUs: $TARGET_CPUS"
log "Log file: $LOG_FILE"
log ""

# Step 1: Generate embeddings for each pretrain epoch
log "======================================"
log "Step 1: Generating Embeddings"
log "======================================"

for epoch in "${EPOCHS[@]}"; do
    log ""
    log "--------------------------------------"
    log "Generating embeddings for epoch $epoch"
    log "--------------------------------------"
    
    python src/scripts/4_batch_embedding_ablation.py --epoch $epoch 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully generated embeddings for epoch $epoch"
    else
        log "✗ Failed to generate embeddings for epoch $epoch"
        exit 1
    fi
done

log ""
log "======================================"
log "Step 2: Training GNN Models"
log "======================================"

# Step 2: Train GNN for each pretrain epoch
for epoch in "${EPOCHS[@]}"; do
    log ""
    log "--------------------------------------"
    log "Training GNN with epoch $epoch embeddings"
    log "--------------------------------------"
    
    python src/scripts/5_gnn_ablation.py \
        --epoch $epoch \
        --source_cpus $SOURCE_CPUS \
        --target_cpus $TARGET_CPUS 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully trained GNN for epoch $epoch"
    else
        log "✗ Failed to train GNN for epoch $epoch"
        exit 1
    fi
done

log ""
log "======================================"
log "Step 3: Aggregating Results"
log "======================================"

# Step 3: Aggregate results
python src/scripts/aggregate_ablation_results.py 2>&1 | tee -a "$LOG_FILE"

if [ $? -eq 0 ]; then
    log "✓ Successfully aggregated results"
else
    log "✗ Failed to aggregate results"
    exit 1
fi

log ""
log "======================================"
log "Ablation Experiments Completed!"
log "======================================"
log "Results saved in: outputs/results/ablation_study/"
log "Log file saved to: $LOG_FILE"
log ""
