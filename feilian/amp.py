"""
Feilian Automatic Mixed Precision (AMP) Module
===============================================

A cross-platform automatic mixed precision implementation that works consistently
across CUDA, MPS, and CPU devices with unified API and smart fallbacks. This module
provides comprehensive AMP support with device-specific optimizations and graceful
degradation on unsupported platforms.

Key Features:
- Cross-platform AMP with unified API
- Device-specific optimization strategies
- Smart fallbacks for unsupported platforms
- Gradient scaling with overflow detection
- Configurable precision modes (FP16, BF16, FP32, Mixed)
- Automatic batch size scaling for memory efficiency
- State management and checkpointing support

Platform Support:
- CUDA: Native torch.cuda.amp integration with full feature support
- CPU: Custom autocast with operation patching for limited mixed precision
- MPS: Limited mixed precision with fallback mechanisms

Precision Modes:
- FP16: Half precision for maximum performance on modern GPUs
- BF16: Brain float16 for improved numerical stability
- FP32: Full precision fallback for compatibility
- Mixed: Automatic selection based on device capabilities

AMP Components:
- AMPManager: Central coordinator for all AMP operations
- GradScaler: Cross-platform gradient scaling with overflow detection
- AutocastContext: Device-specific autocast context management
- AMPConfig: Configuration management with validation

Typical Usage:
    >>> amp_manager = AMPManager()
    >>> with amp_manager.autocast():
    ...     output = model(input)
    ...     loss = criterion(output, target)
    >>> success = amp_manager.backward_and_step(loss, optimizer)

Global Interface:
    >>> with feilian.amp.autocast():
    ...     output = model(input)
    >>> success = feilian.amp.backward_and_step(loss, optimizer)

Author: Feilian Development Team
Version: 1.0.0
"""

import torch
from typing import Optional, Dict, Any, Union, List, Set
from contextlib import contextmanager
from functools import wraps
import logging
from dataclasses import dataclass, field
from enum import Enum
import threading

from .device_manager import DeviceManager


logger = logging.getLogger(__name__)


class PrecisionMode(Enum):
    """Supported precision modes."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    MIXED = "mixed"


@dataclass
class AMPConfig:
    """Configuration for automatic mixed precision."""

    enabled: bool = True
    precision_mode: PrecisionMode = PrecisionMode.FP16
    init_scale: float = 2.0**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    min_scale: float = 1e-4
    max_scale: float = 2.0**24
    enable_caching: bool = True
    autocast_cpu_ops: Set[str] = field(
        default_factory=lambda: {
            "conv1d",
            "conv2d",
            "conv3d",
            "bmm",
            "mm",
            "mv",
            "linear",
            "addmm",
            "addmv",
            "addr",
            "matmul",
            "einsum",
        }
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.init_scale < self.min_scale:
            self.init_scale = self.min_scale
        if self.init_scale > self.max_scale:
            self.init_scale = self.max_scale


class GradScaler:
    """Cross-platform gradient scaler with device-specific optimizations."""

    def __init__(
        self,
        device: torch.device,
        init_scale: float = 2.0**16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        min_scale: float = 1e-4,
        max_scale: float = 2.0**24,
    ):
        self.device = device
        self._scale = torch.tensor(init_scale, dtype=torch.float32, device=device)
        self._growth_tracker = 0
        self._inf_has_been_found = torch.tensor(0.0, dtype=torch.float32, device=device)

        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self.max_scale = max_scale

        self._lock = threading.RLock()

        # Device-specific optimizations
        if device.type == "cuda":
            self._cuda_scaler = torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        else:
            self._cuda_scaler = None

    def scale(
        self, outputs: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Scale loss or gradients."""
        if self._cuda_scaler and self.device.type == "cuda":
            return self._cuda_scaler.scale(outputs)

        if isinstance(outputs, torch.Tensor):
            return outputs * self._scale
        else:
            return [output * self._scale for output in outputs]

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients in optimizer."""
        if self._cuda_scaler and self.device.type == "cuda":
            self._cuda_scaler.unscale_(optimizer)
            return

        with self._lock:
            inv_scale = 1.0 / self._scale
            for param_group in optimizer.param_groups:
                for param in param_group["params"]:
                    if param.grad is not None:
                        param.grad.data.mul_(inv_scale)

    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step optimizer with gradient scaling."""
        if self._cuda_scaler and self.device.type == "cuda":
            # Use CUDA's built-in scaler
            old_scale = self._cuda_scaler.get_scale()
            self._cuda_scaler.step(optimizer)
            self._cuda_scaler.update()
            new_scale = self._cuda_scaler.get_scale()
            return new_scale >= old_scale  # True if no overflow

        with self._lock:
            # Check for inf/nan in gradients
            has_inf = self._check_inf_gradients(optimizer)

            if has_inf:
                # Skip optimizer step and reduce scale
                self._scale.mul_(self.backoff_factor)
                self._scale.clamp_(min=self.min_scale)
                self._growth_tracker = 0
                return False
            else:
                # Unscale gradients
                inv_scale = 1.0 / self._scale
                for param_group in optimizer.param_groups:
                    for param in param_group["params"]:
                        if param.grad is not None:
                            param.grad.data.mul_(inv_scale)

                # Step optimizer
                optimizer.step()

                # Update scale
                self._growth_tracker += 1
                if self._growth_tracker >= self.growth_interval:
                    self._scale.mul_(self.growth_factor)
                    self._scale.clamp_(max=self.max_scale)
                    self._growth_tracker = 0

                return True

    def update(self) -> None:
        """Update the scale factor."""
        if self._cuda_scaler and self.device.type == "cuda":
            self._cuda_scaler.update()

    def get_scale(self) -> float:
        """Get current scale factor."""
        if self._cuda_scaler and self.device.type == "cuda":
            return self._cuda_scaler.get_scale()
        return self._scale.item()

    def set_scale(self, scale: float) -> None:
        """Set scale factor."""
        with self._lock:
            self._scale.fill_(scale)
            if self._cuda_scaler:
                # We can't directly set CUDA scaler scale, so we create a new one
                self._cuda_scaler = torch.cuda.amp.GradScaler(init_scale=scale)

    def _check_inf_gradients(self, optimizer: torch.optim.Optimizer) -> bool:
        """Check if any gradients contain inf or nan."""
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
                        return True
        return False

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary."""
        if self._cuda_scaler and self.device.type == "cuda":
            return self._cuda_scaler.state_dict()

        return {
            "scale": self._scale,
            "growth_tracker": self._growth_tracker,
            "inf_has_been_found": self._inf_has_been_found,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        if self._cuda_scaler and self.device.type == "cuda":
            self._cuda_scaler.load_state_dict(state_dict)
            return

        self._scale = state_dict["scale"].to(self.device)
        self._growth_tracker = state_dict.get("growth_tracker", 0)
        self._inf_has_been_found = state_dict.get(
            "inf_has_been_found", torch.tensor(0.0, device=self.device)
        )


class AutocastContext:
    """Cross-platform autocast context manager."""

    def __init__(
        self,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        enabled: bool = True,
        cache_enabled: Optional[bool] = None,
    ):
        self.device = device
        self.enabled = enabled
        self.dtype = dtype or self._get_default_dtype(device)
        self.cache_enabled = cache_enabled

        # Device-specific context managers
        self._contexts = []

        if enabled:
            if device.type == "cuda" and torch.cuda.is_available():
                # Use CUDA's built-in autocast
                self._contexts.append(
                    torch.cuda.amp.autocast(
                        enabled=True, dtype=self.dtype, cache_enabled=cache_enabled
                    )
                )
            elif device.type == "cpu":
                # Use CPU autocast if available (PyTorch >= 1.10)
                if hasattr(torch, "autocast") and hasattr(torch.autocast, "__call__"):
                    self._contexts.append(
                        torch.autocast(
                            device_type="cpu",
                            dtype=self.dtype,
                            enabled=True,
                            cache_enabled=cache_enabled,
                        )
                    )
                else:
                    # Fallback for older PyTorch versions
                    self._contexts.append(self._cpu_autocast_fallback())
            elif device.type == "mps":
                # MPS autocast support (limited)
                if hasattr(torch, "autocast"):
                    try:
                        self._contexts.append(
                            torch.autocast(
                                device_type="cpu",  # MPS often uses CPU autocast
                                dtype=torch.float16,
                                enabled=True,
                                cache_enabled=cache_enabled,
                            )
                        )
                    except Exception:
                        # Fallback to manual precision control
                        self._contexts.append(self._mps_autocast_fallback())
                else:
                    self._contexts.append(self._mps_autocast_fallback())

    def _get_default_dtype(self, device: torch.device) -> torch.dtype:
        """Get default dtype for autocast on the given device."""
        if device.type == "cuda":
            return torch.float16
        elif device.type == "cpu":
            return torch.bfloat16 if torch.cpu.is_bf16_supported() else torch.float16
        elif device.type == "mps":
            return torch.float16
        else:
            return torch.float16

    @contextmanager
    def _cpu_autocast_fallback(self):
        """Fallback autocast implementation for CPU."""
        if not self.enabled:
            yield
            return

        # Override specific operations to use lower precision
        def autocast_wrapper(op):
            @wraps(op)
            def wrapper(*args, **kwargs):
                # Convert float32 tensors to target dtype for compute-heavy ops
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.Tensor) and arg.dtype == torch.float32:
                        new_args.append(arg.to(self.dtype))
                    else:
                        new_args.append(arg)

                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                        new_kwargs[k] = v.to(self.dtype)
                    else:
                        new_kwargs[k] = v

                result = op(*new_args, **new_kwargs)
                return result

            return wrapper

        # Patch common operations
        ops_to_patch = ["mm", "bmm", "conv1d", "conv2d", "linear"]
        original_ops = {}

        try:
            for op_name in ops_to_patch:
                if hasattr(torch, op_name):
                    original_ops[op_name] = getattr(torch, op_name)
                    setattr(torch, op_name, autocast_wrapper(original_ops[op_name]))
                if hasattr(torch.nn.functional, op_name):
                    original_ops[f"F.{op_name}"] = getattr(torch.nn.functional, op_name)
                    setattr(
                        torch.nn.functional,
                        op_name,
                        autocast_wrapper(original_ops[f"F.{op_name}"]),
                    )

            yield
        finally:
            # Restore original operations
            for op_name in ops_to_patch:
                if op_name in original_ops:
                    setattr(torch, op_name, original_ops[op_name])
                if f"F.{op_name}" in original_ops:
                    setattr(torch.nn.functional, op_name, original_ops[f"F.{op_name}"])

    @contextmanager
    def _mps_autocast_fallback(self):
        """Fallback autocast implementation for MPS."""
        if not self.enabled:
            yield
            return

        # MPS has limited mixed precision support
        # This is a simple fallback that doesn't change precision
        logger.warning("MPS autocast fallback: limited mixed precision support")
        yield

    def __enter__(self):
        """Enter autocast context."""
        for ctx in self._contexts:
            ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit autocast context."""
        # Exit in reverse order
        for ctx in reversed(self._contexts):
            ctx.__exit__(exc_type, exc_val, exc_tb)


class AMPManager:
    """Main automatic mixed precision manager."""

    def __init__(self, device_manager: DeviceManager = None, config: AMPConfig = None):
        self.device_manager = device_manager or DeviceManager()
        self.config = config or AMPConfig()
        self._scalers: Dict[str, GradScaler] = {}
        self._lock = threading.RLock()

        # Initialize scalers for available devices
        self._init_scalers()

    def _init_scalers(self) -> None:
        """Initialize gradient scalers for available devices."""
        available_devices = self.device_manager.get_available_devices()

        for device in available_devices:
            device_key = str(device)

            if device.type in ["cuda", "mps"] or self.config.enabled:
                self._scalers[device_key] = GradScaler(
                    device=device,
                    init_scale=self.config.init_scale,
                    growth_factor=self.config.growth_factor,
                    backoff_factor=self.config.backoff_factor,
                    growth_interval=self.config.growth_interval,
                    min_scale=self.config.min_scale,
                    max_scale=self.config.max_scale,
                )

    def get_scaler(
        self, device: Union[str, torch.device] = "auto"
    ) -> Optional[GradScaler]:
        """Get gradient scaler for a device."""
        if not self.config.enabled:
            return None

        if isinstance(device, str):
            if device == "auto":
                target_device = self.device_manager.get_device("auto")
            else:
                target_device = self.device_manager.get_device(device)
        else:
            target_device = device

        device_key = str(target_device)

        with self._lock:
            if device_key not in self._scalers and self.config.enabled:
                self._scalers[device_key] = GradScaler(
                    device=target_device,
                    init_scale=self.config.init_scale,
                    growth_factor=self.config.growth_factor,
                    backoff_factor=self.config.backoff_factor,
                    growth_interval=self.config.growth_interval,
                    min_scale=self.config.min_scale,
                    max_scale=self.config.max_scale,
                )

            return self._scalers.get(device_key)

    def autocast(
        self,
        device: Union[str, torch.device] = "auto",
        dtype: Optional[torch.dtype] = None,
        enabled: Optional[bool] = None,
    ) -> AutocastContext:
        """Create autocast context for a device."""
        if enabled is None:
            enabled = self.config.enabled

        if isinstance(device, str):
            if device == "auto":
                target_device = self.device_manager.get_device("auto")
            else:
                target_device = self.device_manager.get_device(device)
        else:
            target_device = device

        # Determine dtype based on device and config
        if dtype is None:
            if self.config.precision_mode == PrecisionMode.FP16:
                dtype = torch.float16
            elif self.config.precision_mode == PrecisionMode.BF16:
                dtype = torch.bfloat16
            elif self.config.precision_mode == PrecisionMode.FP32:
                dtype = torch.float32
                enabled = False  # No point in autocasting to fp32
            else:  # MIXED
                if target_device.type == "cuda":
                    dtype = torch.float16
                elif target_device.type == "cpu":
                    dtype = (
                        torch.bfloat16
                        if torch.cpu.is_bf16_supported()
                        else torch.float16
                    )
                else:  # MPS
                    dtype = torch.float16

        return AutocastContext(
            device=target_device,
            dtype=dtype,
            enabled=enabled,
            cache_enabled=self.config.enable_caching,
        )

    def scale_loss(
        self, loss: torch.Tensor, device: Union[str, torch.device] = "auto"
    ) -> torch.Tensor:
        """Scale loss for mixed precision training."""
        scaler = self.get_scaler(device)
        if scaler:
            return scaler.scale(loss)
        return loss

    def backward_and_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        device: Union[str, torch.device] = "auto",
        max_norm: Optional[float] = None,
    ) -> bool:
        """Perform backward pass and optimizer step with AMP."""
        scaler = self.get_scaler(device)

        if scaler is None:
            # Standard training without AMP
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for group in optimizer.param_groups for p in group["params"]],
                    max_norm,
                )
            optimizer.step()
            return True

        # Scale loss and backward
        scaled_loss = scaler.scale(loss)
        scaled_loss.backward()

        # Unscale gradients for clipping
        if max_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                max_norm,
            )

        # Step with scaling
        step_successful = scaler.step(optimizer)
        scaler.update()

        return step_successful

    def get_effective_batch_size(
        self, base_batch_size: int, device: Union[str, torch.device] = "auto"
    ) -> int:
        """Get effective batch size considering AMP scaling."""
        if not self.config.enabled:
            return base_batch_size

        # With mixed precision, we can often use larger batch sizes
        if isinstance(device, str):
            target_device = self.device_manager.get_device(device)
        else:
            target_device = device

        if target_device.type in ["cuda", "mps"]:
            # Can typically use 1.5-2x larger batch size with mixed precision
            return int(base_batch_size * 1.6)

        return base_batch_size

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for all scalers."""
        with self._lock:
            return {
                device_key: scaler.state_dict()
                for device_key, scaler in self._scalers.items()
            }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary for all scalers."""
        with self._lock:
            for device_key, scaler_state in state_dict.items():
                if device_key in self._scalers:
                    self._scalers[device_key].load_state_dict(scaler_state)


# Global AMP manager instance
_global_amp_manager: Optional[AMPManager] = None
_amp_lock = threading.RLock()


def get_amp_manager() -> AMPManager:
    """Get the global AMP manager instance."""
    global _global_amp_manager

    with _amp_lock:
        if _global_amp_manager is None:
            _global_amp_manager = AMPManager()
        return _global_amp_manager


def configure_amp(config: AMPConfig) -> None:
    """Configure the global AMP manager."""
    global _global_amp_manager

    with _amp_lock:
        _global_amp_manager = AMPManager(config=config)


@contextmanager
def autocast(
    device: Union[str, torch.device] = "auto",
    dtype: Optional[torch.dtype] = None,
    enabled: Optional[bool] = None,
):
    """
    Convenience context manager for automatic mixed precision.

    Args:
        device: Target device ('auto', 'cpu', 'cuda', 'mps', or torch.device)
        dtype: Target dtype for mixed precision (None for automatic)
        enabled: Whether to enable AMP (None to use global config)

    Example:
        with feilian.amp.autocast():
            output = model(input)

        with feilian.amp.autocast('cuda', torch.float16):
            loss = criterion(output, target)
    """
    manager = get_amp_manager()
    with manager.autocast(device, dtype, enabled) as ctx:
        yield ctx


def scale_loss(
    loss: torch.Tensor, device: Union[str, torch.device] = "auto"
) -> torch.Tensor:
    """
    Convenience function to scale loss for mixed precision training.

    Args:
        loss: Loss tensor to scale
        device: Target device

    Returns:
        Scaled loss tensor
    """
    return get_amp_manager().scale_loss(loss, device)


def backward_and_step(
    loss: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: Union[str, torch.device] = "auto",
    max_norm: Optional[float] = None,
) -> bool:
    """
    Convenience function for backward pass and optimizer step with AMP.

    Args:
        loss: Loss tensor
        optimizer: Optimizer instance
        device: Target device
        max_norm: Maximum norm for gradient clipping

    Returns:
        True if optimizer step was successful (no overflow)
    """
    return get_amp_manager().backward_and_step(loss, optimizer, device, max_norm)


def get_scaler(device: Union[str, torch.device] = "auto") -> Optional[GradScaler]:
    """
    Get gradient scaler for a device.

    Args:
        device: Target device

    Returns:
        GradScaler instance or None if AMP is disabled
    """
    return get_amp_manager().get_scaler(device)
