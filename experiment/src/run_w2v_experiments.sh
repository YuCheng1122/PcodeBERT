#!/bin/bash

# W2V Experiments Automation Script
# This script runs embedding generation and GNN training for different W2V models

set -e  # Exit on error

BASE_DIR="/home/tommy/Project/PcodeBERT"
cd "$BASE_DIR"

# Define models to test
MODELS=("fasttext" "cbow" "skipgram")

# Source and target CPUs
SOURCE_CPUS="x86_64"
TARGET_CPUS="ARM PPC MIPS Intel"

# Create log directory and file
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/w2v_experiment_${TIMESTAMP}.log"

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# log "======================================"
# log "Starting W2V Experiments"
# log "======================================"
# log "Models: ${MODELS[@]}"
# log "Source CPU: $SOURCE_CPUS"
# log "Target CPUs: $TARGET_CPUS"
# log "Log file: $LOG_FILE"
# log ""

# # Step 1: Generate embeddings for each W2V model
# log "======================================"
# log "Step 1: Generating Embeddings"
# log "======================================"

# python experiment/src/embedding/batch_embedding_w2v.py 2>&1 | tee -a "$LOG_FILE"

# if [ $? -eq 0 ]; then
#     log "✓ Successfully generated embeddings for all models"
# else
#     log "✗ Failed to generate embeddings"
#     exit 1
# fi

log ""
log "======================================"
log "Step 2: Training GNN Models"
log "======================================"

# Step 2: Train GNN for each W2V model
for model in "${MODELS[@]}"; do
    log ""
    log "--------------------------------------"
    log "Training GNN with $model embeddings"
    log "--------------------------------------"
    
    python experiment/src/gnn/gnn_w2v_training.py \
        --model $model \
        --source_cpus $SOURCE_CPUS \
        --target_cpus $TARGET_CPUS 2>&1 | tee -a "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        log "✓ Successfully trained GNN for $model"
    else
        log "✗ Failed to train GNN for $model"
        exit 1
    fi
done

log ""
log "======================================"
log "Step 3: Summary Results"
log "======================================"

# Create summary report
SUMMARY_DIR="outputs/results/w2v_summary"
mkdir -p "$SUMMARY_DIR"
SUMMARY_FILE="$SUMMARY_DIR/experiment_summary_${TIMESTAMP}.txt"

log "Creating experiment summary..."

{
    echo "======================================"
    echo "W2V Experiments Summary"
    echo "======================================"
    echo "Timestamp: $(date)"
    echo "Models tested: ${MODELS[@]}"
    echo "Source CPU: $SOURCE_CPUS"
    echo "Target CPUs: $TARGET_CPUS"
    echo ""
    
    echo "Results directories:"
    for model in "${MODELS[@]}"; do
        result_dir="outputs/results/w2v_${model}"
        if [ -d "$result_dir" ]; then
            echo "  - $model: $result_dir"
            latest_summary=$(ls -t "$result_dir"/summary_*.json 2>/dev/null | head -1)
            if [ -f "$latest_summary" ]; then
                echo "    Latest summary: $(basename "$latest_summary")"
            fi
        fi
    done
    
    echo ""
    echo "Model directories:"
    for model in "${MODELS[@]}"; do
        model_dir="outputs/models/GNN/models_${model}"
        if [ -d "$model_dir" ]; then
            echo "  - $model: $model_dir"
            model_count=$(ls -1 "$model_dir"/*.pt 2>/dev/null | wc -l)
            echo "    Saved models: $model_count"
        fi
    done
    
    echo ""
    echo "Embedding data directories:"
    for model in "${MODELS[@]}"; do
        embed_dir="outputs/data/GNN/gpickle_merged_adjusted_filtered_${model}"
        if [ -d "$embed_dir" ]; then
            echo "  - $model: $embed_dir"
            stats_file="$embed_dir/processing_stats_${model}.json"
            if [ -f "$stats_file" ]; then
                echo "    Statistics: $(basename "$stats_file")"
            fi
        fi
    done
    
} > "$SUMMARY_FILE"

log "Summary report saved to: $SUMMARY_FILE"

log ""
log "======================================"
log "W2V Experiments Completed!"
log "======================================"
log "Results saved in:"
for model in "${MODELS[@]}"; do
    log "  - outputs/results/w2v_${model}/"
done
log "Models saved in:"
for model in "${MODELS[@]}"; do
    log "  - outputs/models/GNN/models_${model}/"
done
log "Log file saved to: $LOG_FILE"
log "Summary report: $SUMMARY_FILE"
log ""

# Display quick stats
log "Quick Statistics:"
for model in "${MODELS[@]}"; do
    stats_file="outputs/data/GNN/gpickle_merged_adjusted_filtered_${model}/processing_stats_${model}.json"
    if [ -f "$stats_file" ]; then
        processed=$(python3 -c "import json; data=json.load(open('$stats_file')); print(f\"Processed: {data.get('processed_files', 0)}, Nodes: {data.get('total_nodes', 0)}, Dim: {data.get('embedding_dim', 0)}\")")
        log "  - $model: $processed"
    fi
done
