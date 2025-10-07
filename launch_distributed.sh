#!/bin/bash

# Feilian-GPU Distributed Training Launch Script
# Provides convenient commands for launching distributed training
# with different configurations.

set -e

# Default parameters
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-1000}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
CHAN_MULTI=${CHAN_MULTI:-20}
MAX_LEVEL=${MAX_LEVEL:-6}
ACTIVATION=${ACTIVATION:-ReLU}
DATA_PATH=${DATA_PATH:-"raw_data/wind3D/idealized/"}
MODEL_DIR=${MODEL_DIR:-"./models"}
DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND:-auto}

# Function to display usage information
usage() {
    echo "Feilian-GPU Distributed Training Launch Script"
    echo "=============================================="
    echo ""
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  single-node-4gpu    Launch single node with 4 GPUs"
    echo "  single-node-8gpu    Launch single node with 8 GPUs"  
    echo "  multi-node          Launch multi-node training"
    echo "  test               Test distributed setup without training"
    echo "  help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  SEED               Random seed (default: 42)"
    echo "  BATCH_SIZE         Batch size per GPU (default: 8)"
    echo "  NUM_EPOCHS         Number of epochs (default: 1000)"
    echo "  LEARNING_RATE      Learning rate (default: 1e-3)"
    echo "  CHAN_MULTI         Channel multiplier (default: 20)"
    echo "  MAX_LEVEL          Network depth (default: 6)" 
    echo "  ACTIVATION         Activation function (default: ReLU)"
    echo "  DATA_PATH          Path to training data (default: raw_data/wind3D/idealized/)"
    echo "  MODEL_DIR          Model save directory (default: ./models)"
    echo "  DISTRIBUTED_BACKEND Backend for distributed training (default: auto)"
    echo ""
    echo "Multi-node specific variables:"
    echo "  NNODES             Number of nodes (default: 2)"
    echo "  NPROC_PER_NODE     GPUs per node (default: 4)"
    echo "  MASTER_ADDR        Master node address"
    echo "  MASTER_PORT        Master node port (default: 29500)"
    echo "  NODE_RANK          Current node rank (0 for master)"
    echo ""
    echo "Examples:"
    echo "  # Single node, 4 GPUs, small batch"
    echo "  BATCH_SIZE=4 $0 single-node-4gpu"
    echo ""
    echo "  # Single node, 8 GPUs, custom data path"
    echo "  DATA_PATH=/path/to/data $0 single-node-8gpu"
    echo ""
    echo "  # Multi-node training (run on master node)"
    echo "  MASTER_ADDR=192.168.1.100 NODE_RANK=0 $0 multi-node"
    echo ""
    echo "  # Multi-node training (run on worker node)"
    echo "  MASTER_ADDR=192.168.1.100 NODE_RANK=1 $0 multi-node"
}

# Function to check if torchrun is available
check_torchrun() {
    if ! command -v torchrun &> /dev/null; then
        echo "ERROR: torchrun is not available. Please install PyTorch >= 1.9.0"
        echo "       pip install torch>=1.9.0"
        exit 1
    fi
}

# Function to validate data path
check_data_path() {
    if [ ! -d "$DATA_PATH" ]; then
        echo "WARNING: Data path '$DATA_PATH' does not exist"
        echo "         Make sure your data is available before training"
    fi
}

# Function to create model directory
create_model_dir() {
    mkdir -p "$MODEL_DIR"
    echo "Model directory: $MODEL_DIR"
}

# Common training arguments
get_training_args() {
    echo "$SEED \
        --data-path $DATA_PATH \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE \
        --num-epochs $NUM_EPOCHS \
        --chan-multi $CHAN_MULTI \
        --max-level $MAX_LEVEL \
        --activation $ACTIVATION \
        --model-dir $MODEL_DIR \
        --distributed \
        --distributed-backend $DISTRIBUTED_BACKEND \
        --save-checkpoints \
        --mixed-precision \
        --verbose"
}

# Single node with 4 GPUs
single_node_4gpu() {
    echo "Starting single-node distributed training with 4 GPUs"
    echo "======================================================"
    echo "Seed: $SEED"
    echo "Batch size per GPU: $BATCH_SIZE"
    echo "Total batch size: $((BATCH_SIZE * 4))"
    echo "Epochs: $NUM_EPOCHS"
    echo "Data path: $DATA_PATH"
    echo "Model directory: $MODEL_DIR"
    echo ""
    
    create_model_dir
    check_data_path
    
    torchrun --nproc_per_node=4 \
             --standalone \
             feilian_main.py $(get_training_args)
}

# Single node with 8 GPUs
single_node_8gpu() {
    echo "Starting single-node distributed training with 8 GPUs"
    echo "======================================================"
    echo "Seed: $SEED"
    echo "Batch size per GPU: $BATCH_SIZE"
    echo "Total batch size: $((BATCH_SIZE * 8))"
    echo "Epochs: $NUM_EPOCHS"
    echo "Data path: $DATA_PATH"
    echo "Model directory: $MODEL_DIR"
    echo ""
    
    create_model_dir
    check_data_path
    
    torchrun --nproc_per_node=8 \
             --standalone \
             feilian_main.py $(get_training_args)
}

# Multi-node training
multi_node() {
    # Default multi-node settings
    NNODES=${NNODES:-2}
    NPROC_PER_NODE=${NPROC_PER_NODE:-4}
    MASTER_PORT=${MASTER_PORT:-29500}
    
    if [ -z "$MASTER_ADDR" ]; then
        echo "ERROR: MASTER_ADDR environment variable is required for multi-node training"
        echo "       Set MASTER_ADDR to the IP address of the master node"
        exit 1
    fi
    
    if [ -z "$NODE_RANK" ]; then
        echo "ERROR: NODE_RANK environment variable is required for multi-node training"
        echo "       Set NODE_RANK to 0 for master node, 1+ for worker nodes"
        exit 1
    fi
    
    echo "Starting multi-node distributed training"
    echo "========================================"
    echo "Nodes: $NNODES"
    echo "GPUs per node: $NPROC_PER_NODE"
    echo "Total GPUs: $((NNODES * NPROC_PER_NODE))"
    echo "Master address: $MASTER_ADDR:$MASTER_PORT"
    echo "Current node rank: $NODE_RANK"
    echo "Batch size per GPU: $BATCH_SIZE"
    echo "Total batch size: $((BATCH_SIZE * NNODES * NPROC_PER_NODE))"
    echo ""
    
    create_model_dir
    check_data_path
    
    torchrun --nnodes=$NNODES \
             --nproc_per_node=$NPROC_PER_NODE \
             --node_rank=$NODE_RANK \
             --master_addr=$MASTER_ADDR \
             --master_port=$MASTER_PORT \
             feilian_main.py $(get_training_args)
}

# Test distributed setup
test_distributed() {
    echo "Testing distributed setup..."
    echo "============================"
    
    python -c "
from feilian.distributed import setup_distributed_training, distributed_context
import torch

print('Testing distributed setup...')
try:
    with distributed_context() as dist_info:
        print(f'Distributed info: {dist_info}')
        if dist_info.is_distributed:
            print(f'Successfully initialized distributed training')
            print(f'Rank: {dist_info.rank}')
            print(f'World size: {dist_info.world_size}')
            print(f'Backend: {dist_info.backend}')
            print(f'Device: {dist_info.device}')
        else:
            print('Running in single-process mode (no RANK/WORLD_SIZE env vars)')
        
        # Test CUDA/MPS availability
        print(f'CUDA available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'CUDA device count: {torch.cuda.device_count()}')
        print(f'MPS available: {torch.backends.mps.is_available() if hasattr(torch.backends, \"mps\") else False}')
        
except Exception as e:
    print(f'Error during distributed setup: {e}')
    import traceback
    traceback.print_exc()
"
}

# Main script logic
main() {
    check_torchrun
    
    case "${1:-help}" in
        single-node-4gpu)
            single_node_4gpu
            ;;
        single-node-8gpu) 
            single_node_8gpu
            ;;
        multi-node)
            multi_node
            ;;
        test)
            test_distributed
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo "ERROR: Unknown command '$1'"
            echo ""
            usage
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"