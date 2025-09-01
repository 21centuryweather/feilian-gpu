# ============================================================================
# Core Module Imports
# ============================================================================
# Import fundamental data processing and neural network components
from .data_formatter import DataFormatter
from .neural_network import (
    FeilianNet,
    CompressionLayer,
    SwitchLayer,
    ExpansionLayer,
    train_network_model_with_adam,
    predict_with_model,
    save_model_state,
    save_checkpoint,
    load_checkpoint,
)

# Cross-platform device management for CUDA, MPS (Apple Silicon), and CPU
from .device_manager import (
    DeviceManager,
    get_device_manager,
    get_best_device,
    print_available_devices,
)

# ============================================================================
# NetCDF Data Loading (Optional Dependency)
# ============================================================================
# NetCDF4 support for 3D wind speed data loading and processing
# Falls back gracefully if netCDF4 is not installed
# NetCDF support - only import if needed (import functions are called directly)
# from .netcdf_loader import ... - imports will be done on-demand

# ============================================================================
# Performance Optimization Modules
# ============================================================================
# Comprehensive benchmarking suite for performance analysis across devices
from .benchmark import BenchmarkRunner, ModelFactory, BenchmarkResult

# Advanced memory management with device-specific optimizations
from .memory_manager import (
    MemoryManager,  # Main memory coordinator
    get_memory_manager,  # Global memory manager instance
    allocate,  # Unified tensor allocation
    free,  # Explicit memory deallocation
    get_memory_stats,  # Memory usage statistics
    optimize_memory,  # Memory optimization and cleanup
)

# Automatic Mixed Precision (AMP) with cross-platform support
from .amp import (
    AMPManager,  # Central AMP coordinator
    get_amp_manager,  # Global AMP manager instance
    autocast,  # Cross-platform autocast context
    PrecisionMode,  # Precision mode enumeration
    AMPConfig,  # AMP configuration dataclass
    scale_loss,  # Loss scaling for mixed precision
    backward_and_step,  # Combined backward pass and optimizer step
)

# Gradient checkpointing for memory-efficient training
from .checkpoint import (
    wrap_with_checkpointing,  # Apply checkpointing to models
    CheckpointConfig,  # Checkpointing configuration
    CheckpointPolicy,  # Policy enumeration
    get_transformer_config,  # Transformer-optimized config
    get_cnn_config,  # CNN-optimized config
    get_memory_optimized_config,  # Memory-efficient config
    analyze_checkpointing_opportunities,  # Analyze model for checkpointing
    estimate_memory_usage,  # Estimate layer memory usage
)

# ============================================================================
# Convenience Module Aliases
# ============================================================================
# Provide convenient access to submodules for advanced usage
from . import benchmark  # feilian.benchmark.BenchmarkRunner()
from . import memory_manager as memory  # feilian.memory.allocate()
from . import amp as mixed_precision  # feilian.mixed_precision.autocast()
from . import checkpoint  # feilian.checkpoint.wrap_with_checkpointing()

# Version information
__version__ = "1.0.0"

__all__ = [
    # Core functionality
    "DataFormatter",
    "FeilianNet",
    "CompressionLayer",
    "SwitchLayer",
    "ExpansionLayer",
    "train_network_model_with_adam",
    "predict_with_model",
    "save_model_state",
    "save_checkpoint",
    "load_checkpoint",
    "DeviceManager",
    "get_device_manager",
    "get_best_device",
    "print_available_devices",
    # Benchmarking
    "BenchmarkRunner",
    "ModelFactory",
    "BenchmarkResult",
    "benchmark",
    # Memory management
    "MemoryManager",
    "get_memory_manager",
    "allocate",
    "free",
    "get_memory_stats",
    "optimize_memory",
    "memory",
    # Automatic mixed precision
    "AMPManager",
    "get_amp_manager",
    "autocast",
    "PrecisionMode",
    "AMPConfig",
    "scale_loss",
    "backward_and_step",
    "mixed_precision",
    # Gradient checkpointing
    "wrap_with_checkpointing",
    "CheckpointConfig",
    "CheckpointPolicy",
    "get_transformer_config",
    "get_cnn_config",
    "get_memory_optimized_config",
    "analyze_checkpointing_opportunities",
    "estimate_memory_usage",
    "checkpoint",
]
