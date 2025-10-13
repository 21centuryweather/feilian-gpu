#!/usr/bin/env python3
"""
SLURM launcher for Feilian distributed training on Setonix.
This script sets up the distributed environment variables and launches feilian_main.py.
"""

import os
import sys


def get_first_node(nodelist):
    """Extract the first node from SLURM nodelist (same logic as get_master.py)."""
    first_nodelist = nodelist.split(',')[0]
    
    if '[' in first_nodelist:
        a = first_nodelist.split('[')
        first_node = a[0] + a[1].split('-')[0]
    else:
        first_node = first_nodelist
    
    return first_node


def setup_distributed_env():
    """Set up distributed training environment variables from SLURM."""
    # Get master address from SLURM_NODELIST
    if 'SLURM_NODELIST' in os.environ:
        master_addr = get_first_node(os.environ['SLURM_NODELIST'])
        os.environ['MASTER_ADDR'] = master_addr
        print(f"Setting MASTER_ADDR={master_addr}")
    else:
        print("Warning: SLURM_NODELIST not found, using localhost")
        os.environ['MASTER_ADDR'] = 'localhost'
    
    # Set master port
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    
    # Set distributed training variables from SLURM
    if 'SLURM_NPROCS' in os.environ:
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NPROCS']
    
    if 'SLURM_PROCID' in os.environ:
        os.environ['RANK'] = os.environ['SLURM_PROCID']
    
    if 'SLURM_LOCALID' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        
        # For multi-node, don't restrict HIP devices too early
        # Let the device manager handle GPU selection based on LOCAL_RANK
        # Only set if not already set by the batch script
        if 'HIP_VISIBLE_DEVICES' not in os.environ:
            os.environ['HIP_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
        if 'ROCR_VISIBLE_DEVICES' not in os.environ:
            os.environ['ROCR_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
    
    print(f"Distributed setup:")
    print(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    print(f"  MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    print(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'unset')}")
    print(f"  RANK: {os.environ.get('RANK', 'unset')}")
    print(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'unset')}")
    print(f"  HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', 'unset')}")


if __name__ == '__main__':
    # Setup distributed environment
    setup_distributed_env()
    
    # Import and run feilian_main
    print("Launching feilian_main.py...")
    
    # Modify sys.argv to remove this script name, so feilian_main gets the right arguments
    sys.argv = ['feilian_main.py'] + sys.argv[1:]
    
    # Import and run feilian_main
    import feilian_main
    feilian_main.main()
