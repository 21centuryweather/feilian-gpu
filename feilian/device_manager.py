"""
Device Management Module for Cross-Platform GPU Support
======================================================

This module provides comprehensive device management and optimization capabilities
across different hardware platforms including Apple Silicon (MPS), NVIDIA CUDA,
and CPU fallback with intelligent device selection and performance optimization.

Key Features:
- Automatic device detection and selection
- Cross-platform GPU support (CUDA, MPS, CPU)
- Device-specific optimizations and memory management
- Intelligent DataLoader configuration
- Performance monitoring and cache management
- Multi-GPU support with DataParallel

Supported Platforms:
- Apple Silicon (M1/M2/M3): Metal Performance Shaders (MPS)
- NVIDIA GPUs: CUDA with mixed precision and multi-GPU
- Intel/AMD CPUs: Optimized threading and memory usage

Typical Usage:
    >>> device_manager = DeviceManager("auto")
    >>> device_manager.print_device_info()
    >>> model = device_manager.optimize_model(model)
    >>> dataloader = device_manager.create_dataloader(dataset, batch_size=32)

Author: Feilian Development Team
Version: 1.0.0
"""

import torch
import logging
import platform
from typing import Optional, Union
from .distributed import setup_distributed_training, DistributedInfo, wrap_model_for_ddp, create_distributed_sampler

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# Device Management Classes
# ============================================================================


class DeviceManager:
    """
    Manages device selection and optimization across different hardware platforms.
    Supports Apple Silicon MPS, NVIDIA CUDA, and CPU fallback.
    """

    def __init__(
        self, 
        device_preference: str = "auto", 
        force_cpu: bool = False,
        enable_distributed: bool = False,
        distributed_backend: Optional[str] = None
    ):
        """
        Initialize the device manager with optional distributed training support.

        Args:
            device_preference: One of "auto", "mps", "cuda", "cpu"
            force_cpu: Force CPU usage regardless of available hardware
            enable_distributed: Enable distributed data parallel training
            distributed_backend: Distributed backend ("nccl", "gloo", "auto", or None)
        """
        self.device_preference = device_preference.lower()
        self.force_cpu = force_cpu
        self.enable_distributed = enable_distributed
        
        # Initialize distributed training if enabled
        if enable_distributed:
            self.distributed_info = setup_distributed_training(distributed_backend)
            logger.info(f"Distributed training enabled: {self.distributed_info}")
        else:
            self.distributed_info = DistributedInfo()
        
        # Select device (may be overridden by distributed setup)
        self.device = self._select_device()
        self.is_apple_silicon = self._is_apple_silicon()

        logger.info(f"Device Manager initialized with device: {self.device}")
        logger.info(f"Platform: {platform.system()} {platform.machine()}")
        if self.distributed_info.is_distributed:
            logger.info(f"Running in distributed mode: rank {self.distributed_info.rank} of {self.distributed_info.world_size}")

    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def _select_device(self) -> torch.device:
        """
        Select the best available device based on preference and availability.
        In distributed mode, uses device from distributed info if available.

        Returns:
            torch.device: The selected device
        """
        # Use distributed device if available
        if self.distributed_info.is_distributed and self.distributed_info.device:
            logger.info(f"Using distributed device: {self.distributed_info.device}")
            return self.distributed_info.device
            
        if self.force_cpu:
            logger.info("Forcing CPU usage as requested")
            return torch.device("cpu")

        if self.device_preference == "cpu":
            logger.info("CPU selected by preference")
            return torch.device("cpu")

        # Check device availability and select based on preference
        if self.device_preference == "auto":
            return self._auto_select_device()
        elif self.device_preference == "mps":
            return self._select_mps()
        elif self.device_preference == "cuda":
            return self._select_cuda()
        else:
            logger.warning(
                f"Unknown device preference: {self.device_preference}, falling back to auto"
            )
            return self._auto_select_device()

    def _auto_select_device(self) -> torch.device:
        """Automatically select the best available device."""
        # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
        if self._is_mps_available():
            logger.info("Auto-selected MPS (Apple Silicon GPU)")
            return torch.device("mps")
        elif self._is_cuda_available():
            cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            gpu_name = torch.cuda.get_device_name()
            logger.info(f"Auto-selected CUDA device: {cuda_device} ({gpu_name})")
            return cuda_device
        else:
            logger.info("Auto-selected CPU (no GPU available)")
            return torch.device("cpu")

    def _select_mps(self) -> torch.device:
        """Select MPS device if available."""
        if self._is_mps_available():
            logger.info("MPS selected and available")
            return torch.device("mps")
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            return torch.device("cpu")

    def _select_cuda(self) -> torch.device:
        """Select CUDA device if available."""
        if self._is_cuda_available():
            cuda_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            gpu_name = torch.cuda.get_device_name()
            logger.info(f"CUDA selected: {cuda_device} ({gpu_name})")
            return cuda_device
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            return torch.device("cpu")

    def _is_mps_available(self) -> bool:
        """Check if MPS is available."""
        try:
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception as e:
            logger.debug(f"MPS check failed: {e}")
            return False

    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            return torch.cuda.is_available() and torch.cuda.device_count() > 0
        except Exception as e:
            logger.debug(f"CUDA check failed: {e}")
            return False

    def get_device_info(self) -> dict:
        """
        Get comprehensive device information.

        Returns:
            dict: Device information including type, memory, etc.
        """
        info = {
            "device": str(self.device),
            "device_type": self.device.type,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "is_apple_silicon": self.is_apple_silicon,
        }

        if self.device.type == "cuda":
            info.update(
                {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(),
                    "cuda_version": torch.version.cuda,
                    "memory_total": f"{torch.cuda.get_device_properties(self.device).total_memory / 1e9:.1f} GB",
                    "memory_allocated": f"{torch.cuda.memory_allocated(self.device) / 1e9:.1f} GB",
                    "memory_reserved": f"{torch.cuda.memory_reserved(self.device) / 1e9:.1f} GB",
                }
            )
        elif self.device.type == "mps":
            info.update({"mps_available": torch.backends.mps.is_available()})

        return info

    def optimize_model(
        self, 
        model: torch.nn.Module, 
        use_compile: bool = True,
        find_unused_parameters: bool = False
    ) -> torch.nn.Module:
        """
        Optimize model for the selected device with optional distributed training.

        Args:
            model: PyTorch model
            use_compile: Whether to use torch.compile (PyTorch 2.0+)
            find_unused_parameters: For DDP, whether to find unused parameters

        Returns:
            Optimized model (wrapped with DDP if distributed)
        """
        # Move model to device
        model = model.to(self.device)

        # Apply device-specific optimizations
        if self.device.type == "cuda":
            model = self._optimize_for_cuda(model)
        elif self.device.type == "mps":
            model = self._optimize_for_mps(model)

        # Apply torch.compile if available and requested (before DDP wrapping)
        if use_compile and hasattr(torch, "compile") and not self.distributed_info.is_distributed:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        # Wrap with DDP if in distributed mode
        if self.distributed_info.is_distributed:
            model = wrap_model_for_ddp(
                model, 
                self.distributed_info, 
                find_unused_parameters=find_unused_parameters
            )
            logger.info("Model wrapped with DistributedDataParallel")

        return model

    def _optimize_for_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply CUDA-specific optimizations."""
        # Enable mixed precision if supported
        if hasattr(torch.cuda, "amp"):
            logger.info("CUDA AMP (Automatic Mixed Precision) available")

        # Multi-GPU support (only if not using distributed training)
        if torch.cuda.device_count() > 1 and not self.distributed_info.is_distributed:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)
        elif self.distributed_info.is_distributed:
            logger.info("Skipping DataParallel - using DistributedDataParallel instead")

        return model

    def _optimize_for_mps(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply MPS-specific optimizations."""
        # MPS-specific settings
        logger.info("Optimizing for Apple Silicon MPS")
        return model

    def create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        drop_last: bool = False
    ) -> torch.utils.data.DataLoader:
        """
        Create optimized DataLoader for the device with distributed support.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size (will be divided by world_size in distributed mode)
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            drop_last: Whether to drop incomplete batches

        Returns:
            Optimized DataLoader with optional distributed sampler
        """
        if num_workers is None:
            # Auto-select number of workers based on device and platform
            if self.device.type == "cpu":
                num_workers = min(4, torch.get_num_threads())
            elif self.device.type == "mps":
                # MPS works better with fewer workers
                num_workers = 2
            else:  # CUDA
                num_workers = 4

        # Pin memory for GPU devices
        pin_memory = self.device.type in ["cuda", "mps"]
        
        # Create distributed sampler if needed
        sampler = None
        effective_batch_size = batch_size
        
        if self.distributed_info.is_distributed:
            sampler = create_distributed_sampler(
                dataset, 
                self.distributed_info, 
                shuffle=shuffle,
                drop_last=drop_last
            )
            # Adjust batch size for distributed training
            effective_batch_size = batch_size // self.distributed_info.world_size
            shuffle = False  # Shuffling handled by DistributedSampler
            logger.info(f"Using distributed sampler with effective batch size: {effective_batch_size}")

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=effective_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=drop_last
        )

        logger.info(
            f"Created DataLoader with {num_workers} workers, pin_memory={pin_memory}"
        )
        if sampler:
            logger.info("DataLoader uses DistributedSampler")
            
        return dataloader

    def move_to_device(self, tensor_or_model):
        """Move tensor or model to the selected device."""
        return tensor_or_model.to(self.device)

    def clear_cache(self):
        """Clear device cache to free memory."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
        elif self.device.type == "mps":
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
                logger.info("MPS cache cleared")

    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.distributed_info.is_distributed
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.distributed_info.is_main_process
    
    def get_rank(self) -> int:
        """Get the current process rank."""
        return self.distributed_info.rank
    
    def get_world_size(self) -> int:
        """Get the total number of processes."""
        return self.distributed_info.world_size
    
    def barrier(self):
        """Synchronize all processes."""
        if self.distributed_info.is_distributed:
            import torch.distributed as dist
            dist.barrier()
    
    def set_epoch(self, dataloader, epoch: int):
        """Set epoch for distributed sampler."""
        if hasattr(dataloader, 'sampler') and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)
            logger.debug(f"Set epoch {epoch} for distributed sampler")

    def print_device_info(self):
        """Print comprehensive device information."""
        info = self.get_device_info()
        print("\n" + "=" * 50)
        print("DEVICE INFORMATION")
        print("=" * 50)

        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("=" * 50)

        # Distributed information
        if self.distributed_info.is_distributed:
            print("\nDISTRIBUTED TRAINING:")
            print(f"Rank: {self.distributed_info.rank}")
            print(f"World Size: {self.distributed_info.world_size}")
            print(f"Backend: {self.distributed_info.backend}")
            print(f"Local Rank: {self.distributed_info.local_rank}")
            print(f"Main Process: {self.distributed_info.is_main_process}")
            print("=" * 50)

        # Additional availability info
        print("\nDEVICE AVAILABILITY:")
        print(f"CUDA Available: {self._is_cuda_available()}")
        print(f"MPS Available: {self._is_mps_available()}")
        print(f"CPU Cores: {torch.get_num_threads()}")
        print("=" * 50 + "\n")


def get_device_manager(
    device_preference: str = "auto", 
    force_cpu: bool = False,
    enable_distributed: bool = False,
    distributed_backend: Optional[str] = None
) -> DeviceManager:
    """
    Factory function to create a DeviceManager instance.

    Args:
        device_preference: One of "auto", "mps", "cuda", "cpu"
        force_cpu: Force CPU usage
        enable_distributed: Enable distributed data parallel training
        distributed_backend: Distributed backend ("nccl", "gloo", "auto", or None)

    Returns:
        DeviceManager instance
    """
    return DeviceManager(device_preference, force_cpu, enable_distributed, distributed_backend)


# Convenience functions
def get_best_device(preference: str = "auto") -> torch.device:
    """Get the best available device."""
    return get_device_manager(preference).device


def print_available_devices():
    """Print information about all available devices."""
    dm = get_device_manager()
    dm.print_device_info()


if __name__ == "__main__":
    # Test the device manager
    print_available_devices()
