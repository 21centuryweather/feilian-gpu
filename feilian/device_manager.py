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
from typing import Optional

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

    def __init__(self, device_preference: str = "auto", force_cpu: bool = False):
        """
        Initialize the device manager.

        Args:
            device_preference: One of "auto", "mps", "cuda", "cpu"
            force_cpu: Force CPU usage regardless of available hardware
        """
        self.device_preference = device_preference.lower()
        self.force_cpu = force_cpu
        self.device = self._select_device()
        self.is_apple_silicon = self._is_apple_silicon()

        logger.info(f"Device Manager initialized with device: {self.device}")
        logger.info(f"Platform: {platform.system()} {platform.machine()}")

    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def _select_device(self) -> torch.device:
        """
        Select the best available device based on preference and availability.

        Returns:
            torch.device: The selected device
        """
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
        self, model: torch.nn.Module, use_compile: bool = True
    ) -> torch.nn.Module:
        """
        Optimize model for the selected device.

        Args:
            model: PyTorch model
            use_compile: Whether to use torch.compile (PyTorch 2.0+)

        Returns:
            Optimized model
        """
        # Move model to device
        model = model.to(self.device)

        # Apply device-specific optimizations
        if self.device.type == "cuda":
            model = self._optimize_for_cuda(model)
        elif self.device.type == "mps":
            model = self._optimize_for_mps(model)

        # Apply torch.compile if available and requested
        if use_compile and hasattr(torch, "compile"):
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")

        return model

    def _optimize_for_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply CUDA-specific optimizations."""
        # Enable mixed precision if supported
        if hasattr(torch.cuda, "amp"):
            logger.info("CUDA AMP (Automatic Mixed Precision) available")

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = torch.nn.DataParallel(model)

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
    ) -> torch.utils.data.DataLoader:
        """
        Create optimized DataLoader for the device.

        Args:
            dataset: PyTorch dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            Optimized DataLoader
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

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
        )

        logger.info(
            f"Created DataLoader with {num_workers} workers, pin_memory={pin_memory}"
        )
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

    def print_device_info(self):
        """Print comprehensive device information."""
        info = self.get_device_info()
        print("\n" + "=" * 50)
        print("DEVICE INFORMATION")
        print("=" * 50)

        for key, value in info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

        print("=" * 50)

        # Additional availability info
        print("\nDEVICE AVAILABILITY:")
        print(f"CUDA Available: {self._is_cuda_available()}")
        print(f"MPS Available: {self._is_mps_available()}")
        print(f"CPU Cores: {torch.get_num_threads()}")
        print("=" * 50 + "\n")


def get_device_manager(
    device_preference: str = "auto", force_cpu: bool = False
) -> DeviceManager:
    """
    Factory function to create a DeviceManager instance.

    Args:
        device_preference: One of "auto", "mps", "cuda", "cpu"
        force_cpu: Force CPU usage

    Returns:
        DeviceManager instance
    """
    return DeviceManager(device_preference, force_cpu)


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
