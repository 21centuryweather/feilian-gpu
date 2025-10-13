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
    different hardware platforms. It properly handles cases where the process group
    is already initialized (e.g., when using torchrun).
    
    Args:
        backend: Distributed backend ("nccl", "gloo", "auto", or None for auto)
        init_method: Process group initialization method
        timeout_minutes: Timeout for process group initialization
        
    Returns:
        DistributedInfo: Information about the distributed setup
        
    Raises:
        RuntimeError: If distributed initialization fails
    """
    # Check if process group is already initialized
    if dist.is_initialized():
        logger.info("Distributed process group already initialized - using existing setup")
        return _get_existing_distributed_info()
    
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


def _get_existing_distributed_info() -> DistributedInfo:
    """Get distributed information from an already initialized process group."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = dist.get_backend()
    
    # Get device for local rank
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
    
    logger.info(f"Using existing distributed setup: rank {rank} of {world_size}, backend {backend}")
    
    return distributed_info


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


# ============================================================================
# Spawn-based Distributed Training (Alternative to Torchrun)
# ============================================================================

import socket
import torch.multiprocessing as mp
from functools import partial

def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def spawn_distributed_training(
    training_fn,
    world_size: Optional[int] = None,
    master_addr: str = "127.0.0.1",
    master_port: Optional[int] = None,
    backend: Optional[str] = None,
    **kwargs
):
    """
    Launch distributed training using torch.multiprocessing.spawn.
    
    This is a production-ready alternative to torchrun that can be called 
    directly from Python. It automatically handles GPU assignment and uses 
    the most compatible backend for the hardware.
    
    Args:
        training_fn: Function to run on each process. Should accept (rank, world_size, **kwargs)
        world_size: Number of processes to spawn (default: torch.cuda.device_count())
        master_addr: Master node address for process coordination
        master_port: Master node port (default: auto-select free port)
        backend: Distributed backend ("gloo", "nccl", or None for auto-select)
        **kwargs: Additional arguments passed to training_fn
        
    Example:
        def train_worker(rank, world_size, data_path, batch_size):
            # Your training code here
            pass
            
        spawn_distributed_training(
            train_worker,
            world_size=4,
            data_path="data/",
            batch_size=32
        )
    """
    if world_size is None:
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if world_size == 0:
            world_size = 1
            
    if world_size == 1:
        logger.info("World size is 1, calling training function directly (no spawn needed)")
        # Set up minimal environment for single-GPU case
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(find_free_port())
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        logger.info("World size is 1, running single-process training")
        training_fn(0, 1, **kwargs)
        return
        
    if master_port is None:
        master_port = find_free_port()
        
    # Auto-select backend if not specified
    if backend is None:
        # NCCL doesn't seem to work very well on gadi so using Gloo here. 
        # Will need to switch back to NCCL for Setonix.
        backend = "gloo"
        
    logger.info(f"Launching {world_size} processes with spawn method")
    logger.info(f"Master: {master_addr}:{master_port}, Backend: {backend}")
    
    # Set environment variables for the spawned processes
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    
    try:
        mp.spawn(
            _spawn_worker_wrapper,
            args=(training_fn, world_size, backend, kwargs),
            nprocs=world_size,
            join=True,
            daemon=False,
            start_method='spawn'
        )
        logger.info("All spawn processes completed successfully")
    except Exception as e:
        logger.error(f"Spawn distributed training failed: {e}")
        raise


def _spawn_worker_wrapper(rank: int, training_fn, world_size: int, backend: str, kwargs: Dict[str, Any]):
    """
    Worker wrapper function for spawn processes.
    
    This function sets up the distributed environment for each spawned process
    and then calls the user-provided training function.
    """
    try:
        # Set up process-specific environment
        setup_process_distributed(rank, world_size, backend)
        
        # Call user training function
        training_fn(rank, world_size, **kwargs)
        
    except Exception as e:
        logger.error(f"Process {rank} failed: {e}")
        raise
    finally:
        # Clean up distributed resources
        if dist.is_initialized():
            dist.destroy_process_group()


def setup_process_distributed(
    rank: int, 
    world_size: int, 
    backend: Optional[str] = None,
    force_cpu: bool = False
) -> DistributedInfo:
    """
    Set up distributed training for a single spawned process.
    
    This function is called within each spawned process to initialize distributed
    training. It handles device assignment and process group initialization.
    
    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: Distributed backend (default: "gloo")
        force_cpu: Force CPU training even if CUDA is available
        
    Returns:
        DistributedInfo: Information about the distributed setup
    """
    # Set environment variables that would normally be set by torchrun
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    if backend is None:
        backend = "gloo"  # Most reliable for spawn
    
    logger.info(f"Process {rank}: Initializing distributed training")
    
    try:
        # Check if process group is already initialized
        if dist.is_initialized():
            logger.warning(f"Process {rank}: Process group already initialized, destroying existing one")
            dist.destroy_process_group()
        
        # Initialize process group
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        
        # Set device for this process
        if torch.cuda.is_available() and not force_cpu:
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
            logger.info(f"Process {rank}: Using GPU {device}")
        else:
            device = torch.device("cpu")
            logger.info(f"Process {rank}: Using CPU")
        
        distributed_info = DistributedInfo(
            is_distributed=True,
            rank=rank,
            local_rank=rank,  # For single-node spawn, rank == local_rank
            world_size=world_size,
            backend=backend,
            is_main_process=(rank == 0),
            device=device
        )
        
        if rank == 0:
            logger.info(f"Successfully initialized spawn distributed training with {world_size} processes")
            logger.info(f"Backend: {backend}, Device: {device}")
        
        return distributed_info
        
    except Exception as e:
        logger.error(f"Process {rank}: Failed to initialize distributed training: {e}")
        raise RuntimeError(f"Process {rank}: Distributed initialization failed: {e}") from e


# ============================================================================
# High-Level Integration with Existing Training Infrastructure
# ============================================================================

def spawn_feilian_training(
    data_path: str,
    world_size: Optional[int] = None,
    batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    save_checkpoints: bool = False,
    checkpoint_interval: int = 5,
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    High-level function to launch Feilian training using spawn method.
    
    This function provides a direct alternative to torchrun-based training
    and integrates with the existing Feilian training infrastructure.
    
    Args:
        data_path: Path to training data
        world_size: Number of GPUs to use (default: all available)
        batch_size: Batch size per GPU
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        save_checkpoints: Whether to save checkpoints
        checkpoint_interval: Epochs between checkpoint saves
        model_config: Model configuration dictionary
        **kwargs: Additional training arguments
        
    Example:
        # Direct spawn-based training
        spawn_feilian_training(
            data_path="raw_data/wind3D/idealized/",
            world_size=4,
            batch_size=8,
            num_epochs=10,
            learning_rate=1e-3
        )
    """
    spawn_distributed_training(
        _feilian_training_worker,
        world_size=world_size,
        data_path=data_path,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_checkpoints=save_checkpoints,
        checkpoint_interval=checkpoint_interval,
        model_config=model_config or {},
        **kwargs
    )


def _feilian_training_worker(
    rank: int,
    world_size: int,
    data_path: str,
    batch_size: int = 8,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    save_checkpoints: bool = False,
    checkpoint_interval: int = 5,
    model_config: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """
    Worker function for Feilian training in spawn mode.
    
    This function integrates with the existing Feilian training infrastructure
    while providing distributed training capabilities via spawn.
    """
    import torch.optim as optim
    from torch.utils.data import DataLoader
    
    try:
        # Set up distributed training for this process
        distributed_info = setup_process_distributed(rank, world_size)
        
        if distributed_info.is_main_process:
            logger.info(f"Starting Feilian training on {world_size} processes")
            logger.info(f"Data path: {data_path}")
            logger.info(f"Batch size per process: {batch_size}")
            logger.info(f"Total effective batch size: {batch_size * world_size}")
        
        # Try to load actual Feilian data and model
        try:
            from feilian import DataFormatter, FeilianNet
            
            # Load and format data
            data_formatter = DataFormatter()
            dataset = data_formatter.load_data(data_path)
            
            # Create model
            if model_config:
                model = FeilianNet(**model_config)
            else:
                model = FeilianNet()
                
            logger.info(f"Process {rank}: Loaded Feilian data and model")
            
        except (ImportError, Exception) as e:
            logger.warning(f"Process {rank}: Could not load Feilian components: {e}")
            logger.info(f"Process {rank}: Using synthetic data for testing")
            
            # Fallback to synthetic dataset for testing
            import torch.utils.data as data
            
            class SyntheticDataset(data.Dataset):
                def __init__(self, size=1000, input_dim=100):
                    self.size = size
                    self.data = torch.randn(size, input_dim)
                    self.targets = torch.randn(size, 1)
                    
                def __len__(self):
                    return self.size
                    
                def __getitem__(self, idx):
                    return self.data[idx], self.targets[idx]
            
            dataset = SyntheticDataset()
            
            # Create a simple test model
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )
        
        # Create distributed sampler
        sampler = create_distributed_sampler(dataset, distributed_info, shuffle=True)
        
        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            pin_memory=True,
            num_workers=2
        )
        
        # Move model to appropriate device and wrap with DDP
        model = model.to(distributed_info.device)
        model = wrap_model_for_ddp(model, distributed_info)
        
        # Create optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        if distributed_info.is_main_process:
            logger.info("Starting training loop...")
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            if sampler and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
                
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, (data, target) in enumerate(dataloader):
                data = data.to(distributed_info.device)
                target = target.to(distributed_info.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss across all processes
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            metrics = {"loss": avg_loss}
            averaged_metrics = all_reduce_metrics(metrics, distributed_info)
            
            if distributed_info.is_main_process:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {averaged_metrics['loss']:.6f}")
            
            # Save checkpoint if requested
            if save_checkpoints and (epoch + 1) % checkpoint_interval == 0:
                if distributed_info.is_main_process:
                    checkpoint_path = f"checkpoint_spawn_epoch_{epoch+1}.pth"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': averaged_metrics['loss'],
                        'distributed_info': {
                            'world_size': world_size,
                            'backend': distributed_info.backend,
                            'method': 'spawn'
                        }
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        barrier_and_print(f"Training completed successfully on {world_size} processes!", distributed_info)
        
    except Exception as e:
        logger.error(f"Process {rank} training failed: {e}")
        raise


