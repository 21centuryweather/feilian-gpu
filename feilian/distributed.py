"""
Distributed Training Utilities for Cross-Platform Support
==========================================================

This module provides distributed training capabilities that work seamlessly
across different hardware platforms including Apple Silicon (MPS), NVIDIA CUDA,
and multi-node configurations. It integrates with the existing DeviceManager
to maintain all advanced features while adding distributed data parallel support.

Key Features:
- Cross-platform backend selection (NCCL for CUDA, Gloo for others)
- Automatic process group initialization with environment variable detection
- Rank-aware logging and device assignment
- Integration with existing DeviceManager and mixed precision training
- Support for both single-node multi-GPU and multi-node distributed training

Typical Usage:
    >>> from feilian.distributed import setup_distributed_training
    >>> distributed_info = setup_distributed_training()
    >>> if distributed_info.is_distributed:
    ...     print(f"Rank {distributed_info.rank} of {distributed_info.world_size}")

Launch Examples:
    # Single node, 4 GPUs
    torchrun --nproc_per_node=4 feilian_main.py --distributed
    
    # Multi-node (2 nodes, 4 GPUs each)
    torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=123 \
             --rdzv_backend=c10d --rdzv_endpoint=master_node:29500 \
             feilian_main.py --distributed

Author: Feilian Development Team
Version: 1.0.0
"""

import os
import logging
import platform
from typing import Optional, Dict, Any
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.data.sampler import Sampler

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Distributed Information Classes
# ============================================================================

@dataclass
class DistributedInfo:
    """Container for distributed training information."""
    is_distributed: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "nccl"
    is_main_process: bool = True
    device: Optional[torch.device] = None

# ============================================================================
# Core Distributed Functions
# ============================================================================

def setup_distributed_training(
    backend: Optional[str] = None,
    init_method: str = "env://",
    timeout_minutes: int = 30
) -> DistributedInfo:
    """
    Initialize distributed training with cross-platform backend selection.
    
    This function automatically detects distributed training environment variables
    and initializes the process group with appropriate backend selection for
    different hardware platforms.
    
    Args:
        backend: Distributed backend ("nccl", "gloo", "auto", or None for auto)
        init_method: Process group initialization method
        timeout_minutes: Timeout for process group initialization
        
    Returns:
        DistributedInfo: Information about the distributed setup
        
    Raises:
        RuntimeError: If distributed initialization fails
    """
    # Check if we're in a distributed environment
    if not _is_distributed_environment():
        logger.info("Not in distributed environment - running single process")
        return DistributedInfo(
            is_distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True
        )
    
    # Get distributed environment information
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))  
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    # Auto-select backend if not specified
    if backend is None or backend == "auto":
        backend = _select_backend()
    
    logger.info(f"Initializing distributed training: rank={rank}, world_size={world_size}, backend={backend}")
    
    try:
        # Initialize process group with timeout
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.distributed_c10d.default_pg_timeout * timeout_minutes
        )
        
        # Verify initialization
        if not dist.is_initialized():
            raise RuntimeError("Failed to initialize distributed process group")
            
        # Set device for local rank
        device = _get_device_for_rank(local_rank, backend)
        
        distributed_info = DistributedInfo(
            is_distributed=True,
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            is_main_process=(rank == 0),
            device=device
        )
        
        if rank == 0:
            logger.info(f"Successfully initialized distributed training with {world_size} processes")
            logger.info(f"Backend: {backend}, Device: {device}")
        
        return distributed_info
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        raise RuntimeError(f"Distributed initialization failed: {e}") from e


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            logger.info("Distributed process group cleaned up")
        except Exception as e:
            logger.warning(f"Error during distributed cleanup: {e}")


def create_distributed_sampler(
    dataset,
    distributed_info: DistributedInfo,
    shuffle: bool = True,
    drop_last: bool = False
) -> Optional[Sampler]:
    """
    Create distributed sampler if in distributed mode.
    
    Args:
        dataset: PyTorch dataset
        distributed_info: Distributed training information
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop incomplete batches
        
    Returns:
        DistributedSampler if distributed, None otherwise
    """
    if not distributed_info.is_distributed:
        return None
        
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed_info.world_size,
        rank=distributed_info.rank,
        shuffle=shuffle,
        drop_last=drop_last
    )
    
    logger.info(f"Created distributed sampler for rank {distributed_info.rank}")
    return sampler


def wrap_model_for_ddp(
    model: torch.nn.Module,
    distributed_info: DistributedInfo,
    find_unused_parameters: bool = False,
    gradient_as_bucket_view: bool = False
) -> torch.nn.Module:
    """
    Wrap model with DistributedDataParallel if in distributed mode.
    
    Args:
        model: PyTorch model
        distributed_info: Distributed training information
        find_unused_parameters: Whether to find unused parameters
        gradient_as_bucket_view: Optimization for gradient synchronization
        
    Returns:
        DDP-wrapped model if distributed, original model otherwise
    """
    if not distributed_info.is_distributed:
        return model
        
    # Move model to appropriate device
    if distributed_info.device:
        model = model.to(distributed_info.device)
    
    # Wrap with DDP
    ddp_model = DDP(
        model,
        device_ids=[distributed_info.local_rank] if distributed_info.device and distributed_info.device.type == "cuda" else None,
        output_device=distributed_info.local_rank if distributed_info.device and distributed_info.device.type == "cuda" else None,
        find_unused_parameters=find_unused_parameters,
        gradient_as_bucket_view=gradient_as_bucket_view
    )
    
    logger.info(f"Wrapped model with DDP for rank {distributed_info.rank}")
    return ddp_model


def all_reduce_metrics(metrics: Dict[str, float], distributed_info: DistributedInfo) -> Dict[str, float]:
    """
    All-reduce metrics across all processes for accurate averaging.
    
    Args:
        metrics: Dictionary of metric name to value
        distributed_info: Distributed training information
        
    Returns:
        Dictionary of averaged metrics across all processes
    """
    if not distributed_info.is_distributed:
        return metrics
        
    averaged_metrics = {}
    for name, value in metrics.items():
        tensor = torch.tensor(value, dtype=torch.float32, device=distributed_info.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        averaged_metrics[name] = tensor.item()
        
    return averaged_metrics


def barrier_and_print(message: str, distributed_info: DistributedInfo):
    """
    Print message only from main process after synchronization barrier.
    
    Args:
        message: Message to print
        distributed_info: Distributed training information
    """
    if distributed_info.is_distributed:
        dist.barrier()
        
    if distributed_info.is_main_process:
        print(message)


# ============================================================================
# Helper Functions
# ============================================================================

def _is_distributed_environment() -> bool:
    """Check if running in a distributed training environment."""
    required_vars = ["RANK", "WORLD_SIZE"]
    return all(var in os.environ for var in required_vars)


def _select_backend() -> str:
    """
    Automatically select the best backend for the current platform.
    
    Returns:
        Backend name ("nccl" for CUDA, "gloo" for others)
    """
    # Check if CUDA is available and we're not on Apple Silicon
    if torch.cuda.is_available() and not _is_apple_silicon():
        logger.info("Auto-selected NCCL backend for CUDA")
        return "nccl"
    else:
        logger.info("Auto-selected Gloo backend for non-CUDA or Apple Silicon")
        return "gloo"


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _get_device_for_rank(local_rank: int, backend: str) -> torch.device:
    """
    Get the appropriate device for the given local rank.
    
    Args:
        local_rank: Local rank of the process
        backend: Distributed backend being used
        
    Returns:
        torch.device: Device for this rank
    """
    if backend == "nccl" and torch.cuda.is_available():
        # For NCCL, use CUDA device corresponding to local rank
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        logger.info(f"Set CUDA device {device} for local rank {local_rank}")
        return device
    elif torch.backends.mps.is_available() and hasattr(torch.backends, "mps"):
        # For Apple Silicon, use MPS
        device = torch.device("mps")
        logger.info(f"Using MPS device for local rank {local_rank}")
        return device
    else:
        # Fallback to CPU
        device = torch.device("cpu")
        logger.info(f"Using CPU device for local rank {local_rank}")
        return device


def get_distributed_info() -> DistributedInfo:
    """
    Get current distributed training information.
    
    Returns:
        DistributedInfo with current state
    """
    if not dist.is_initialized():
        return DistributedInfo()
        
    return DistributedInfo(
        is_distributed=True,
        rank=dist.get_rank(),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=dist.get_world_size(),
        backend=dist.get_backend(),
        is_main_process=(dist.get_rank() == 0)
    )


# ============================================================================
# Context Managers
# ============================================================================

class distributed_context:
    """
    Context manager for distributed training setup and cleanup.
    
    Example:
        with distributed_context() as dist_info:
            if dist_info.is_distributed:
                print(f"Running on rank {dist_info.rank}")
            # Training code here...
    """
    
    def __init__(self, backend: Optional[str] = None):
        self.backend = backend
        self.distributed_info = None
    
    def __enter__(self) -> DistributedInfo:
        self.distributed_info = setup_distributed_training(self.backend)
        return self.distributed_info
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()


if __name__ == "__main__":
    # Test distributed setup
    with distributed_context() as dist_info:
        print(f"Distributed info: {dist_info}")
        if dist_info.is_distributed:
            print(f"Rank {dist_info.rank} of {dist_info.world_size} on device {dist_info.device}")
        else:
            print("Running in single-process mode")