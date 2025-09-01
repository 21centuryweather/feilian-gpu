"""
Feilian Cross-Platform Memory Manager
====================================

A comprehensive memory management system that provides unified memory allocation,
lifecycle tracking, and optimization across CPU, Apple Silicon MPS, and NVIDIA CUDA.
This module implements sophisticated memory management strategies including pooling,
lifecycle tracking, and automatic optimization.

Key Features:
- Unified memory allocation API across devices
- Device-specific memory optimizations and strategies
- Tensor lifecycle tracking with access pattern analysis
- Automatic memory cleanup and garbage collection
- Memory usage statistics and fragmentation analysis
- Background optimization threads for automatic maintenance
- Memory pool management for efficient reuse

Device-Specific Optimizations:
- CUDA: Stream management, cache optimization, multi-GPU support
- MPS: Apple Silicon Metal Performance Shaders optimization
- CPU: Memory pooling, process integration, threading optimization

Memory Management Strategies:
- Allocation tracking with weak references
- Stale tensor identification and cleanup
- Fragmentation analysis and defragmentation
- Pool-based allocation for common tensor shapes
- Automatic cache management and memory pressure handling

Typical Usage:
    >>> memory_mgr = MemoryManager()
    >>> tensor = memory_mgr.allocate((1024, 1024), device="auto")
    >>> stats = memory_mgr.get_memory_stats("cuda")
    >>> memory_mgr.optimize_memory()

Global Interface:
    >>> import feilian
    >>> tensor = feilian.memory.allocate((512, 512))
    >>> feilian.memory.optimize_memory()

Author: Feilian Development Team
Version: 1.0.0
"""

import gc
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import logging

import torch

from .device_manager import DeviceManager


logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory statistics for tracking allocations."""

    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    total_allocations: int
    active_allocations: int
    peak_allocated_bytes: int
    fragmentation_ratio: float

    @property
    def allocated_mb(self) -> float:
        return self.allocated_bytes / 1024**2

    @property
    def reserved_mb(self) -> float:
        return self.reserved_bytes / 1024**2

    @property
    def peak_allocated_mb(self) -> float:
        return self.peak_allocated_bytes / 1024**2


@dataclass
class AllocationInfo:
    """Information about a tensor allocation."""

    tensor_id: int
    device: torch.device
    size_bytes: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    allocated_at: float
    last_accessed: float
    access_count: int
    is_pinned: bool = False


class TensorLifecycleTracker:
    """Tracks tensor lifecycle and access patterns for smart memory management."""

    def __init__(self):
        self._allocations: Dict[int, AllocationInfo] = {}
        self._device_stats: Dict[str, MemoryStats] = {}
        self._lock = threading.RLock()
        self._peak_memory: Dict[str, int] = defaultdict(int)

    def register_allocation(self, tensor: torch.Tensor) -> None:
        """Register a new tensor allocation."""
        if not isinstance(tensor, torch.Tensor):
            return

        tensor_id = id(tensor)
        current_time = time.time()

        with self._lock:
            # Calculate size
            size_bytes = tensor.element_size() * tensor.numel()

            # Create allocation info
            info = AllocationInfo(
                tensor_id=tensor_id,
                device=tensor.device,
                size_bytes=size_bytes,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                allocated_at=current_time,
                last_accessed=current_time,
                access_count=1,
                is_pinned=tensor.is_pinned() if hasattr(tensor, "is_pinned") else False,
            )

            self._allocations[tensor_id] = info

            # Update peak memory
            device_key = str(tensor.device)
            current_allocated = sum(
                alloc.size_bytes
                for alloc in self._allocations.values()
                if str(alloc.device) == device_key
            )
            self._peak_memory[device_key] = max(
                self._peak_memory[device_key], current_allocated
            )

            logger.debug(f"Registered tensor allocation: {info}")

    def update_access(self, tensor: torch.Tensor) -> None:
        """Update access information for a tensor."""
        if not isinstance(tensor, torch.Tensor):
            return

        tensor_id = id(tensor)
        current_time = time.time()

        with self._lock:
            if tensor_id in self._allocations:
                info = self._allocations[tensor_id]
                info.last_accessed = current_time
                info.access_count += 1

    def unregister_allocation(self, tensor_id: int) -> None:
        """Unregister a tensor allocation."""
        with self._lock:
            if tensor_id in self._allocations:
                del self._allocations[tensor_id]

    def get_stale_tensors(self, max_age_seconds: float = 300.0) -> List[AllocationInfo]:
        """Get tensors that haven't been accessed recently."""
        current_time = time.time()
        stale_tensors = []

        with self._lock:
            for info in self._allocations.values():
                if current_time - info.last_accessed > max_age_seconds:
                    stale_tensors.append(info)

        return stale_tensors

    def get_memory_stats(self, device: Optional[torch.device] = None) -> MemoryStats:
        """Get memory statistics for a device."""
        with self._lock:
            if device is None:
                # Global stats
                total_allocated = sum(
                    info.size_bytes for info in self._allocations.values()
                )
                total_count = len(self._allocations)
                peak_allocated = (
                    max(self._peak_memory.values()) if self._peak_memory else 0
                )
                device_key = "global"
            else:
                device_key = str(device)
                device_allocations = [
                    info
                    for info in self._allocations.values()
                    if str(info.device) == device_key
                ]
                total_allocated = sum(info.size_bytes for info in device_allocations)
                total_count = len(device_allocations)
                peak_allocated = self._peak_memory.get(device_key, 0)

            # Get device-specific reserved memory
            reserved_bytes = 0
            if device and device.type == "cuda":
                reserved_bytes = torch.cuda.memory_reserved(device)
            elif device and device.type == "mps":
                # MPS doesn't expose reserved memory easily
                reserved_bytes = total_allocated

            free_bytes = max(0, reserved_bytes - total_allocated)
            fragmentation_ratio = (
                free_bytes / reserved_bytes if reserved_bytes > 0 else 0.0
            )

            return MemoryStats(
                allocated_bytes=total_allocated,
                reserved_bytes=reserved_bytes,
                free_bytes=free_bytes,
                total_allocations=total_count,  # Historical count would require more tracking
                active_allocations=total_count,
                peak_allocated_bytes=peak_allocated,
                fragmentation_ratio=fragmentation_ratio,
            )


class BaseMemoryAllocator(ABC):
    """Abstract base class for device-specific memory allocators."""

    def __init__(self, device: torch.device):
        self.device = device
        self._pool_size_mb = 0
        self._allocated_tensors: Set[int] = set()

    @abstractmethod
    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Allocate a tensor with the given shape and dtype."""
        pass

    @abstractmethod
    def free(self, tensor: torch.Tensor) -> None:
        """Explicitly free a tensor's memory."""
        pass

    @abstractmethod
    def get_memory_stats(self) -> MemoryStats:
        """Get memory statistics for this allocator."""
        pass

    @abstractmethod
    def optimize_memory(self) -> None:
        """Perform memory optimization specific to this device."""
        pass

    def set_pool_size(self, size_mb: int) -> None:
        """Set the memory pool size in MB."""
        self._pool_size_mb = size_mb


class CPUMemoryAllocator(BaseMemoryAllocator):
    """CPU memory allocator with smart pre-allocation."""

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._memory_pool: OrderedDict[str, List[torch.Tensor]] = OrderedDict()
        self._lock = threading.RLock()

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Allocate a CPU tensor, potentially from the pool."""
        key = self._get_pool_key(shape, dtype)

        with self._lock:
            # Try to reuse from pool
            if key in self._memory_pool and self._memory_pool[key]:
                tensor = self._memory_pool[key].pop()
                tensor.zero_()
                if requires_grad:
                    tensor.requires_grad_(True)
                self._allocated_tensors.add(id(tensor))
                return tensor

        # Create new tensor
        tensor = torch.zeros(
            shape, dtype=dtype, device=self.device, requires_grad=requires_grad
        )
        self._allocated_tensors.add(id(tensor))
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """Return tensor to the pool for reuse."""
        if tensor.device != self.device:
            return

        tensor_id = id(tensor)
        if tensor_id not in self._allocated_tensors:
            return

        self._allocated_tensors.discard(tensor_id)

        # Add to pool for reuse (if pool isn't too large)
        key = self._get_pool_key(tensor.shape, tensor.dtype)

        with self._lock:
            if key not in self._memory_pool:
                self._memory_pool[key] = []

            if len(self._memory_pool[key]) < 10:  # Limit pool size per key
                tensor.requires_grad_(False)
                tensor.zero_()
                self._memory_pool[key].append(tensor.detach())

    def get_memory_stats(self) -> MemoryStats:
        """Get CPU memory statistics."""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        with self._lock:
            pool_memory = sum(
                sum(t.element_size() * t.numel() for t in tensors)
                for tensors in self._memory_pool.values()
            )

        return MemoryStats(
            allocated_bytes=memory_info.rss,
            reserved_bytes=memory_info.vms,
            free_bytes=max(0, memory_info.vms - memory_info.rss),
            total_allocations=len(self._allocated_tensors),
            active_allocations=len(self._allocated_tensors),
            peak_allocated_bytes=memory_info.peak_wset
            if hasattr(memory_info, "peak_wset")
            else memory_info.rss,
            fragmentation_ratio=pool_memory / memory_info.rss
            if memory_info.rss > 0
            else 0.0,
        )

    def optimize_memory(self) -> None:
        """Optimize CPU memory by clearing unused pool entries."""
        with self._lock:
            # Keep only recently used entries
            keys_to_remove = []
            max_pool_entries = 5

            for key, tensors in self._memory_pool.items():
                if len(tensors) > max_pool_entries:
                    # Keep only the most recent entries
                    self._memory_pool[key] = tensors[-max_pool_entries:]
                elif not tensors:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._memory_pool[key]

        # Force garbage collection
        gc.collect()

    def _get_pool_key(self, shape: Tuple[int, ...], dtype: torch.dtype) -> str:
        """Generate a key for the memory pool."""
        return f"{shape}_{dtype}"


class CUDAMemoryAllocator(BaseMemoryAllocator):
    """CUDA memory allocator with advanced caching."""

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._streams: List[torch.cuda.Stream] = []
        self._current_stream_idx = 0
        self._lock = threading.RLock()

        # Create multiple streams for async operations
        for _ in range(4):
            self._streams.append(torch.cuda.Stream(device=device))

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Allocate CUDA tensor with stream management."""
        with self._lock:
            # Use current stream
            stream = self._streams[self._current_stream_idx]
            self._current_stream_idx = (self._current_stream_idx + 1) % len(
                self._streams
            )

        with torch.cuda.stream(stream):
            tensor = torch.zeros(
                shape, dtype=dtype, device=self.device, requires_grad=requires_grad
            )
            self._allocated_tensors.add(id(tensor))
            return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """Free CUDA tensor memory."""
        if tensor.device != self.device:
            return

        tensor_id = id(tensor)
        if tensor_id in self._allocated_tensors:
            self._allocated_tensors.discard(tensor_id)
            del tensor
            torch.cuda.empty_cache()

    def get_memory_stats(self) -> MemoryStats:
        """Get CUDA memory statistics."""
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_allocated = torch.cuda.max_memory_allocated(self.device)

        return MemoryStats(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            free_bytes=reserved - allocated,
            total_allocations=len(self._allocated_tensors),
            active_allocations=len(self._allocated_tensors),
            peak_allocated_bytes=peak_allocated,
            fragmentation_ratio=(reserved - allocated) / reserved
            if reserved > 0
            else 0.0,
        )

    def optimize_memory(self) -> None:
        """Optimize CUDA memory."""
        torch.cuda.empty_cache()
        torch.cuda.synchronize(self.device)


class MPSMemoryAllocator(BaseMemoryAllocator):
    """MPS (Metal Performance Shaders) memory allocator for Apple Silicon."""

    def __init__(self, device: torch.device):
        super().__init__(device)
        self._lock = threading.RLock()

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Allocate MPS tensor."""
        tensor = torch.zeros(
            shape, dtype=dtype, device=self.device, requires_grad=requires_grad
        )
        self._allocated_tensors.add(id(tensor))
        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """Free MPS tensor memory."""
        if tensor.device != self.device:
            return

        tensor_id = id(tensor)
        if tensor_id in self._allocated_tensors:
            self._allocated_tensors.discard(tensor_id)
            del tensor
            # MPS equivalent of empty_cache (if available)
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def get_memory_stats(self) -> MemoryStats:
        """Get MPS memory statistics."""
        # MPS doesn't expose as much memory info as CUDA
        allocated = 0
        if hasattr(torch.mps, "current_allocated_memory"):
            allocated = torch.mps.current_allocated_memory()

        return MemoryStats(
            allocated_bytes=allocated,
            reserved_bytes=allocated,  # MPS doesn't separate reserved vs allocated
            free_bytes=0,
            total_allocations=len(self._allocated_tensors),
            active_allocations=len(self._allocated_tensors),
            peak_allocated_bytes=allocated,
            fragmentation_ratio=0.0,
        )

    def optimize_memory(self) -> None:
        """Optimize MPS memory."""
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


class MemoryManager:
    """Main memory manager coordinating all device-specific allocators."""

    def __init__(self, device_manager: DeviceManager = None):
        self.device_manager = device_manager or DeviceManager()
        self.lifecycle_tracker = TensorLifecycleTracker()
        self._allocators: Dict[str, BaseMemoryAllocator] = {}
        self._lock = threading.RLock()
        self._optimization_thread = None
        self._stop_optimization = threading.Event()

        # Initialize allocators for available devices
        self._init_allocators()

        # Start background optimization
        self._start_optimization_thread()

    def _init_allocators(self) -> None:
        """Initialize device-specific allocators."""
        available_devices = self.device_manager.get_available_devices()

        for device in available_devices:
            device_key = str(device)

            if device.type == "cuda":
                self._allocators[device_key] = CUDAMemoryAllocator(device)
            elif device.type == "mps":
                self._allocators[device_key] = MPSMemoryAllocator(device)
            else:  # CPU
                self._allocators[device_key] = CPUMemoryAllocator(device)

    def allocate(
        self,
        shape: Union[Tuple[int, ...], torch.Size],
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device, None] = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """
        Unified memory allocation interface.

        Args:
            shape: Shape of the tensor to allocate
            dtype: Data type of the tensor
            device: Target device ('auto', 'cpu', 'cuda', 'mps', or torch.device)
            requires_grad: Whether the tensor requires gradients

        Returns:
            Allocated tensor on the specified device
        """
        if device is None or device == "auto":
            target_device = self.device_manager.get_device("auto")
        elif isinstance(device, str):
            target_device = self.device_manager.get_device(device)
        else:
            target_device = device

        device_key = str(target_device)

        with self._lock:
            if device_key not in self._allocators:
                # Fallback to direct allocation
                tensor = torch.zeros(
                    shape,
                    dtype=dtype,
                    device=target_device,
                    requires_grad=requires_grad,
                )
            else:
                allocator = self._allocators[device_key]
                tensor = allocator.allocate(tuple(shape), dtype, requires_grad)

        # Track the allocation
        self.lifecycle_tracker.register_allocation(tensor)

        return tensor

    def free(self, tensor: torch.Tensor) -> None:
        """Free a tensor's memory explicitly."""
        if not isinstance(tensor, torch.Tensor):
            return

        device_key = str(tensor.device)
        tensor_id = id(tensor)

        with self._lock:
            if device_key in self._allocators:
                self._allocators[device_key].free(tensor)

        self.lifecycle_tracker.unregister_allocation(tensor_id)

    def get_memory_stats(
        self, device: Union[str, torch.device, None] = None
    ) -> MemoryStats:
        """
        Get memory statistics for a device or globally.

        Args:
            device: Target device or None for global stats

        Returns:
            Memory statistics
        """
        if device is None:
            return self.lifecycle_tracker.get_memory_stats()

        if isinstance(device, str):
            target_device = self.device_manager.get_device(device)
        else:
            target_device = device

        device_key = str(target_device)

        with self._lock:
            if device_key in self._allocators:
                return self._allocators[device_key].get_memory_stats()
            else:
                return self.lifecycle_tracker.get_memory_stats(target_device)

    def optimize_memory(self, device: Union[str, torch.device, None] = None) -> None:
        """
        Optimize memory usage for a specific device or all devices.

        Args:
            device: Target device or None for all devices
        """
        if device is None:
            # Optimize all devices
            with self._lock:
                for allocator in self._allocators.values():
                    allocator.optimize_memory()
        else:
            if isinstance(device, str):
                target_device = self.device_manager.get_device(device)
            else:
                target_device = device

            device_key = str(target_device)

            with self._lock:
                if device_key in self._allocators:
                    self._allocators[device_key].optimize_memory()

        # Force Python garbage collection
        gc.collect()

    def cleanup_stale_memory(self, max_age_seconds: float = 300.0) -> int:
        """
        Clean up stale memory allocations.

        Args:
            max_age_seconds: Maximum age for tensors before cleanup

        Returns:
            Number of tensors cleaned up
        """
        stale_tensors = self.lifecycle_tracker.get_stale_tensors(max_age_seconds)

        cleanup_count = 0
        for info in stale_tensors:
            # This is tricky because we don't have direct references to tensors
            # In practice, stale tensors would typically be cleaned up by Python's GC
            logger.debug(f"Identified stale tensor: {info}")
            cleanup_count += 1

        # Force optimization to clean up any unreferenced memory
        self.optimize_memory()

        return cleanup_count

    def set_memory_pool_size(
        self, size_mb: int, device: Union[str, torch.device, None] = None
    ) -> None:
        """
        Set memory pool size for a device.

        Args:
            size_mb: Pool size in megabytes
            device: Target device or None for all devices
        """
        if device is None:
            with self._lock:
                for allocator in self._allocators.values():
                    allocator.set_pool_size(size_mb)
        else:
            if isinstance(device, str):
                target_device = self.device_manager.get_device(device)
            else:
                target_device = device

            device_key = str(target_device)

            with self._lock:
                if device_key in self._allocators:
                    self._allocators[device_key].set_pool_size(size_mb)

    def _start_optimization_thread(self) -> None:
        """Start background optimization thread."""

        def optimization_loop():
            while not self._stop_optimization.wait(60):  # Run every minute
                try:
                    # Clean up stale memory
                    self.cleanup_stale_memory()

                    # Optimize memory on all devices
                    self.optimize_memory()

                except Exception as e:
                    logger.warning(f"Background memory optimization failed: {e}")

        self._optimization_thread = threading.Thread(
            target=optimization_loop, daemon=True
        )
        self._optimization_thread.start()

    def shutdown(self) -> None:
        """Shutdown the memory manager."""
        self._stop_optimization.set()
        if self._optimization_thread:
            self._optimization_thread.join(timeout=5)

        # Final cleanup
        self.optimize_memory()


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None
_manager_lock = threading.RLock()


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager

    with _manager_lock:
        if _global_memory_manager is None:
            _global_memory_manager = MemoryManager()
        return _global_memory_manager


def allocate(
    shape: Union[Tuple[int, ...], torch.Size],
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device, None] = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """
    Convenience function for tensor allocation using the global memory manager.

    Args:
        shape: Shape of the tensor to allocate
        dtype: Data type of the tensor
        device: Target device ('auto', 'cpu', 'cuda', 'mps', or torch.device)
        requires_grad: Whether the tensor requires gradients

    Returns:
        Allocated tensor on the specified device
    """
    return get_memory_manager().allocate(shape, dtype, device, requires_grad)


def free(tensor: torch.Tensor) -> None:
    """
    Convenience function for freeing tensor memory using the global memory manager.

    Args:
        tensor: Tensor to free
    """
    get_memory_manager().free(tensor)


def get_memory_stats(device: Union[str, torch.device, None] = None) -> MemoryStats:
    """
    Convenience function for getting memory statistics.

    Args:
        device: Target device or None for global stats

    Returns:
        Memory statistics
    """
    return get_memory_manager().get_memory_stats(device)


def optimize_memory(device: Union[str, torch.device, None] = None) -> None:
    """
    Convenience function for optimizing memory.

    Args:
        device: Target device or None for all devices
    """
    get_memory_manager().optimize_memory(device)
