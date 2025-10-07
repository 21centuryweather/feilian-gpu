#!/usr/bin/env python3
"""
Distributed Training Demo Script

This script demonstrates distributed training capabilities on a Mac
using CPU-based distributed training with multiple processes.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, '.')

from feilian import FeilianNet, train_network_model_with_adam
from feilian.distributed import setup_distributed_training, cleanup_distributed


def main():
    print("=== Feilian-GPU Distributed Training Demo ===")
    
    # Initialize distributed training
    dist_info = setup_distributed_training()
    
    if dist_info.is_main_process:
        print(f"Running distributed training:")
        print(f"  - Processes: {dist_info.world_size}")
        print(f"  - Backend: {dist_info.backend}")
        print(f"  - Device: {dist_info.device}")
    
    print(f"Process rank {dist_info.rank} starting...")
    
    try:
        # Load test data
        x_train = np.load('test_x_train.npy')
        y_train = np.load('test_y_train.npy')
        
        if dist_info.is_main_process:
            print(f"Loaded data: {x_train.shape} -> {y_train.shape}")
        
        # Create a small model for testing
        model = FeilianNet(
            chan_multi=4,  # Small model for quick testing
            max_level=2,
            activation=nn.ReLU()
        )
        
        if dist_info.is_main_process:
            print(f"Model parameters: {model.count_trainable_parameters():,}")
        
        # Train with distributed support
        print(f"Rank {dist_info.rank}: Starting training...")
        
        trained_model = train_network_model_with_adam(
            model,
            x_train,
            y_train,
            batch_size=2,           # Small batch for demo
            lr=1e-3,
            num_epochs=5,           # Few epochs for demo
            model_dir="./test_models",
            device_preference="cpu", # Force CPU for cross-platform demo
            enable_distributed=True,
            distributed_backend="gloo"  # Gloo works on all platforms
        )
        
        if dist_info.is_main_process:
            print("✓ Distributed training completed successfully!")
            print("✓ Model saved by main process")
        
        print(f"Rank {dist_info.rank}: Training complete")
        
    except Exception as e:
        print(f"Error on rank {dist_info.rank}: {e}")
        raise
    
    finally:
        # Clean up distributed resources
        cleanup_distributed()
        if dist_info.is_main_process:
            print("✓ Distributed cleanup completed")


if __name__ == "__main__":
    main()