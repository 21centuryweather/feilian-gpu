"""
Feilian Gradient Checkpointing Utilities
========================================

Provides flexible gradient checkpointing with customizable policies for memory optimization
in deep neural networks with minimal code changes. This module implements various
checkpointing strategies to reduce memory usage during training while maintaining
computational efficiency.

Key Features:
- Multiple checkpointing policies (every-N, attention-only, memory-based, custom)
- Automatic model analysis for optimal checkpointing placement
- Memory usage estimation and optimization recommendations
- Flexible policy framework for custom strategies
- Integration with PyTorch's gradient checkpointing infrastructure
- Sequential and recursive checkpointing support

Checkpointing Policies:
- EveryNPolicy: Checkpoint every N layers for uniform memory reduction
- AttentionOnlyPolicy: Target attention layers specifically
- HeavyComputePolicy: Focus on computationally expensive operations
- MemoryEfficientPolicy: Memory usage threshold-based checkpointing
- CustomPolicy: User-defined checkpointing logic

Memory Optimization Strategies:
- Layer-wise memory analysis and profiling
- Checkpoint opportunity identification
- Memory savings estimation
- Fragmentation analysis and optimization
- Automatic policy recommendation based on model architecture

Model Integration:
- Transparent model wrapping with minimal code changes
- Recursive and sequential checkpointing modes
- Preservation of model interface and behavior
- Support for complex model architectures and custom layers

Typical Usage:
    >>> config = CheckpointConfig(policy=CheckpointPolicy.EVERY_N, interval=3)
    >>> model = wrap_with_checkpointing(model, config)
    >>> analysis = analyze_checkpointing_opportunities(model)

Predefined Configurations:
    >>> transformer_config = get_transformer_config()
    >>> cnn_config = get_cnn_config()
    >>> memory_config = get_memory_optimized_config(threshold_mb=100)

Author: Feilian Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint_utils
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from abc import ABC, abstractmethod
from functools import wraps
import weakref
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CheckpointPolicy(Enum):
    """Predefined checkpointing policies."""
    NONE = "none"                    # No checkpointing
    EVERY_N = "every_n"             # Checkpoint every N layers
    ATTENTION_ONLY = "attention"    # Checkpoint only attention layers
    HEAVY_COMPUTE = "heavy_compute" # Checkpoint computationally expensive operations
    MEMORY_EFFICIENT = "memory"     # Checkpoint based on memory usage
    CUSTOM = "custom"               # Custom user-defined policy


@dataclass
class CheckpointConfig:
    """Configuration for gradient checkpointing."""
    policy: CheckpointPolicy = CheckpointPolicy.EVERY_N
    interval: int = 2                          # For EVERY_N policy
    memory_threshold_mb: float = 100.0         # For MEMORY_EFFICIENT policy
    preserve_rng_state: bool = True            # Whether to preserve RNG state
    pack_hook_handle: bool = False             # Use pack/unpack hooks for efficiency
    use_reentrant: bool = True                 # Use reentrant checkpointing
    
    # Layer type targeting for specific policies
    attention_layers: Set[type] = None         # Types to checkpoint for ATTENTION_ONLY
    heavy_compute_layers: Set[type] = None     # Types to checkpoint for HEAVY_COMPUTE
    
    def __post_init__(self):
        if self.attention_layers is None:
            self.attention_layers = {
                nn.MultiheadAttention,
                # Add more attention layer types as needed
            }
        
        if self.heavy_compute_layers is None:
            self.heavy_compute_layers = {
                nn.Conv1d, nn.Conv2d, nn.Conv3d,
                nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                nn.Linear, nn.MultiheadAttention,
                nn.LSTM, nn.GRU, nn.RNN
            }


class CheckpointPolicyInterface(ABC):
    """Abstract interface for checkpoint policies."""
    
    @abstractmethod
    def should_checkpoint(self, layer: nn.Module, layer_idx: int, 
                         memory_usage: Optional[float] = None) -> bool:
        """Determine if a layer should be checkpointed."""
        pass
    
    @abstractmethod
    def get_checkpoint_fn(self) -> Callable:
        """Get the appropriate checkpoint function to use."""
        pass


class EveryNPolicy(CheckpointPolicyInterface):
    """Checkpoint every N layers."""
    
    def __init__(self, interval: int = 2, preserve_rng: bool = True, use_reentrant: bool = True):
        self.interval = interval
        self.preserve_rng = preserve_rng
        self.use_reentrant = use_reentrant
    
    def should_checkpoint(self, layer: nn.Module, layer_idx: int, 
                         memory_usage: Optional[float] = None) -> bool:
        return layer_idx % self.interval == 0
    
    def get_checkpoint_fn(self) -> Callable:
        def checkpoint_fn(function, *args, **kwargs):
            return checkpoint_utils.checkpoint(
                function, *args, 
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng,
                **kwargs
            )
        return checkpoint_fn


class AttentionOnlyPolicy(CheckpointPolicyInterface):
    """Checkpoint only attention layers."""
    
    def __init__(self, attention_types: Set[type] = None, preserve_rng: bool = True, 
                 use_reentrant: bool = True):
        self.attention_types = attention_types or {nn.MultiheadAttention}
        self.preserve_rng = preserve_rng
        self.use_reentrant = use_reentrant
    
    def should_checkpoint(self, layer: nn.Module, layer_idx: int, 
                         memory_usage: Optional[float] = None) -> bool:
        return any(isinstance(layer, att_type) for att_type in self.attention_types)
    
    def get_checkpoint_fn(self) -> Callable:
        def checkpoint_fn(function, *args, **kwargs):
            return checkpoint_utils.checkpoint(
                function, *args,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng,
                **kwargs
            )
        return checkpoint_fn


class HeavyComputePolicy(CheckpointPolicyInterface):
    """Checkpoint computationally expensive layers."""
    
    def __init__(self, heavy_types: Set[type] = None, preserve_rng: bool = True,
                 use_reentrant: bool = True):
        self.heavy_types = heavy_types or {
            nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.Linear, nn.MultiheadAttention, nn.LSTM, nn.GRU
        }
        self.preserve_rng = preserve_rng
        self.use_reentrant = use_reentrant
    
    def should_checkpoint(self, layer: nn.Module, layer_idx: int, 
                         memory_usage: Optional[float] = None) -> bool:
        return any(isinstance(layer, heavy_type) for heavy_type in self.heavy_types)
    
    def get_checkpoint_fn(self) -> Callable:
        def checkpoint_fn(function, *args, **kwargs):
            return checkpoint_utils.checkpoint(
                function, *args,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng,
                **kwargs
            )
        return checkpoint_fn


class MemoryEfficientPolicy(CheckpointPolicyInterface):
    """Checkpoint based on memory usage thresholds."""
    
    def __init__(self, memory_threshold_mb: float = 100.0, preserve_rng: bool = True,
                 use_reentrant: bool = True):
        self.memory_threshold_mb = memory_threshold_mb
        self.preserve_rng = preserve_rng
        self.use_reentrant = use_reentrant
    
    def should_checkpoint(self, layer: nn.Module, layer_idx: int, 
                         memory_usage: Optional[float] = None) -> bool:
        if memory_usage is None:
            # Fallback to parameter count heuristic
            param_count = sum(p.numel() for p in layer.parameters())
            # Rough estimate: 4 bytes per parameter (float32)
            estimated_memory_mb = param_count * 4 / (1024 * 1024)
            return estimated_memory_mb > self.memory_threshold_mb
        
        return memory_usage > self.memory_threshold_mb
    
    def get_checkpoint_fn(self) -> Callable:
        def checkpoint_fn(function, *args, **kwargs):
            return checkpoint_utils.checkpoint(
                function, *args,
                use_reentrant=self.use_reentrant,
                preserve_rng_state=self.preserve_rng,
                **kwargs
            )
        return checkpoint_fn


class CustomPolicy(CheckpointPolicyInterface):
    """Custom user-defined policy."""
    
    def __init__(self, should_checkpoint_fn: Callable, checkpoint_fn: Optional[Callable] = None):
        self._should_checkpoint_fn = should_checkpoint_fn
        self._checkpoint_fn = checkpoint_fn or self._default_checkpoint_fn
    
    def should_checkpoint(self, layer: nn.Module, layer_idx: int, 
                         memory_usage: Optional[float] = None) -> bool:
        return self._should_checkpoint_fn(layer, layer_idx, memory_usage)
    
    def get_checkpoint_fn(self) -> Callable:
        return self._checkpoint_fn
    
    def _default_checkpoint_fn(self, function, *args, **kwargs):
        return checkpoint_utils.checkpoint(function, *args, **kwargs)


class CheckpointWrapper(nn.Module):
    """Wrapper that applies checkpointing to a module based on policy."""
    
    def __init__(self, module: nn.Module, policy: CheckpointPolicyInterface,
                 layer_idx: int = 0):
        super().__init__()
        self.module = module
        self.policy = policy
        self.layer_idx = layer_idx
        self._should_checkpoint = policy.should_checkpoint(module, layer_idx)
        self._checkpoint_fn = policy.get_checkpoint_fn()
    
    def forward(self, *args, **kwargs):
        if self._should_checkpoint and self.training:
            return self._checkpoint_fn(self.module, *args, **kwargs)
        else:
            return self.module(*args, **kwargs)
    
    def __getattr__(self, name: str):
        # Delegate attribute access to the wrapped module
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class SequentialCheckpointer(nn.Module):
    """Sequential container that applies checkpointing based on policy."""
    
    def __init__(self, *modules, policy: CheckpointPolicyInterface):
        super().__init__()
        self.policy = policy
        self._modules_dict = nn.ModuleDict()
        
        for i, module in enumerate(modules):
            layer_name = f"layer_{i}"
            if policy.should_checkpoint(module, i):
                wrapped_module = CheckpointWrapper(module, policy, i)
                self._modules_dict[layer_name] = wrapped_module
                logger.debug(f"Checkpointing enabled for layer {i}: {type(module).__name__}")
            else:
                self._modules_dict[layer_name] = module
    
    def forward(self, x):
        for module in self._modules_dict.values():
            x = module(x)
        return x
    
    def __len__(self):
        return len(self._modules_dict)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return list(self._modules_dict.values())[idx]
        return self._modules_dict[idx]


def wrap_with_checkpointing(model: nn.Module, 
                           config: CheckpointConfig = None,
                           policy: CheckpointPolicyInterface = None,
                           recursive: bool = True,
                           layer_filter: Optional[Callable[[nn.Module], bool]] = None) -> nn.Module:
    """
    Wrap a model with gradient checkpointing.
    
    Args:
        model: The model to wrap
        config: Checkpointing configuration
        policy: Custom policy (overrides config.policy)
        recursive: Whether to apply checkpointing recursively to submodules
        layer_filter: Optional filter function to determine which layers to consider
        
    Returns:
        Model with checkpointing applied
    """
    if config is None:
        config = CheckpointConfig()
    
    if policy is None:
        # Create policy based on config
        if config.policy == CheckpointPolicy.EVERY_N:
            policy = EveryNPolicy(
                interval=config.interval,
                preserve_rng=config.preserve_rng_state,
                use_reentrant=config.use_reentrant
            )
        elif config.policy == CheckpointPolicy.ATTENTION_ONLY:
            policy = AttentionOnlyPolicy(
                attention_types=config.attention_layers,
                preserve_rng=config.preserve_rng_state,
                use_reentrant=config.use_reentrant
            )
        elif config.policy == CheckpointPolicy.HEAVY_COMPUTE:
            policy = HeavyComputePolicy(
                heavy_types=config.heavy_compute_layers,
                preserve_rng=config.preserve_rng_state,
                use_reentrant=config.use_reentrant
            )
        elif config.policy == CheckpointPolicy.MEMORY_EFFICIENT:
            policy = MemoryEfficientPolicy(
                memory_threshold_mb=config.memory_threshold_mb,
                preserve_rng=config.preserve_rng_state,
                use_reentrant=config.use_reentrant
            )
        elif config.policy == CheckpointPolicy.NONE:
            return model  # No checkpointing
        else:
            raise ValueError(f"Policy {config.policy} requires custom policy object")
    
    # Apply checkpointing
    if recursive:
        return _wrap_recursive(model, policy, layer_filter)
    else:
        return _wrap_sequential(model, policy, layer_filter)


def _wrap_recursive(model: nn.Module, 
                   policy: CheckpointPolicyInterface,
                   layer_filter: Optional[Callable[[nn.Module], bool]] = None) -> nn.Module:
    """Recursively wrap modules with checkpointing."""
    layer_idx = 0
    
    def _apply_checkpointing(module: nn.Module) -> nn.Module:
        nonlocal layer_idx
        
        # Apply filter if provided
        if layer_filter and not layer_filter(module):
            return module
        
        # Check if this module should be checkpointed
        if policy.should_checkpoint(module, layer_idx):
            logger.debug(f"Applying checkpointing to {type(module).__name__} at layer {layer_idx}")
            wrapped = CheckpointWrapper(module, policy, layer_idx)
            layer_idx += 1
            return wrapped
        
        layer_idx += 1
        
        # Recursively apply to children
        for name, child in module.named_children():
            new_child = _apply_checkpointing(child)
            if new_child is not child:
                setattr(module, name, new_child)
        
        return module
    
    return _apply_checkpointing(model)


def _wrap_sequential(model: nn.Module,
                    policy: CheckpointPolicyInterface,
                    layer_filter: Optional[Callable[[nn.Module], bool]] = None) -> nn.Module:
    """Wrap sequential models with checkpointing."""
    if hasattr(model, '_modules') and isinstance(model._modules, dict):
        # Handle modules with ordered children
        new_modules = []
        for i, (name, module) in enumerate(model._modules.items()):
            if layer_filter and not layer_filter(module):
                new_modules.append(module)
                continue
                
            if policy.should_checkpoint(module, i):
                wrapped = CheckpointWrapper(module, policy, i)
                new_modules.append(wrapped)
                logger.debug(f"Checkpointing enabled for {name}: {type(module).__name__}")
            else:
                new_modules.append(module)
        
        # Create new sequential container
        if isinstance(model, nn.Sequential):
            return nn.Sequential(*new_modules)
        else:
            # For other container types, replace children
            for i, (name, _) in enumerate(model._modules.items()):
                setattr(model, name, new_modules[i])
            return model
    
    # For non-container modules, wrap directly if needed
    if policy.should_checkpoint(model, 0):
        return CheckpointWrapper(model, policy, 0)
    
    return model


def checkpoint_sequential(functions: List[nn.Module], 
                         segments: int,
                         preserve_rng_state: bool = True,
                         use_reentrant: bool = True) -> Callable:
    """
    Create a checkpointed sequential function from a list of modules.
    
    Args:
        functions: List of modules/functions to execute sequentially
        segments: Number of segments to divide the functions into for checkpointing
        preserve_rng_state: Whether to preserve RNG state
        use_reentrant: Whether to use reentrant checkpointing
        
    Returns:
        Checkpointed sequential function
    """
    def sequential_forward(input_tensor):
        def run_function(start_idx, end_idx):
            def forward(x):
                for idx in range(start_idx, end_idx):
                    x = functions[idx](x)
                return x
            return forward
        
        # Divide functions into segments
        segment_size = len(functions) // segments
        x = input_tensor
        
        for i in range(segments):
            start_idx = i * segment_size
            if i == segments - 1:
                end_idx = len(functions)  # Last segment gets remaining functions
            else:
                end_idx = (i + 1) * segment_size
            
            segment_fn = run_function(start_idx, end_idx)
            
            if i < segments - 1:  # Don't checkpoint the last segment
                x = checkpoint_utils.checkpoint(
                    segment_fn, x,
                    preserve_rng_state=preserve_rng_state,
                    use_reentrant=use_reentrant
                )
            else:
                x = segment_fn(x)
        
        return x
    
    return sequential_forward


# Convenience functions and decorators

def checkpoint_function(preserve_rng_state: bool = True, use_reentrant: bool = True):
    """Decorator to checkpoint a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return checkpoint_utils.checkpoint(
                func, *args,
                preserve_rng_state=preserve_rng_state,
                use_reentrant=use_reentrant,
                **kwargs
            )
        return wrapper
    return decorator


def estimate_memory_usage(module: nn.Module) -> float:
    """
    Estimate memory usage of a module in MB.
    
    Args:
        module: Module to estimate memory usage for
        
    Returns:
        Estimated memory usage in MB
    """
    param_memory = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in module.buffers())
    
    # Rough estimate including activations (assume 2x parameter memory for activations)
    total_memory = (param_memory + buffer_memory) * 3
    
    return total_memory / (1024 * 1024)  # Convert to MB


def analyze_checkpointing_opportunities(model: nn.Module, 
                                      memory_threshold_mb: float = 50.0) -> Dict[str, Any]:
    """
    Analyze a model to identify good checkpointing opportunities.
    
    Args:
        model: Model to analyze
        memory_threshold_mb: Memory threshold for recommending checkpointing
        
    Returns:
        Analysis results with recommendations
    """
    results = {
        'total_layers': 0,
        'heavy_layers': [],
        'attention_layers': [],
        'recommended_checkpoints': [],
        'total_memory_estimate_mb': 0.0,
        'potential_savings_mb': 0.0
    }
    
    layer_idx = 0
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue  # Skip container modules
        
        memory_usage = estimate_memory_usage(module)
        results['total_memory_estimate_mb'] += memory_usage
        results['total_layers'] += 1
        
        # Identify attention layers
        if isinstance(module, (nn.MultiheadAttention,)):
            results['attention_layers'].append({
                'name': name,
                'type': type(module).__name__,
                'memory_mb': memory_usage,
                'layer_idx': layer_idx
            })
        
        # Identify heavy compute layers
        heavy_types = {
            nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d,
            nn.Linear, nn.MultiheadAttention, nn.LSTM, nn.GRU
        }
        
        if any(isinstance(module, t) for t in heavy_types):
            results['heavy_layers'].append({
                'name': name,
                'type': type(module).__name__,
                'memory_mb': memory_usage,
                'layer_idx': layer_idx
            })
        
        # Recommend checkpointing for high-memory layers
        if memory_usage > memory_threshold_mb:
            results['recommended_checkpoints'].append({
                'name': name,
                'type': type(module).__name__,
                'memory_mb': memory_usage,
                'layer_idx': layer_idx,
                'reason': 'High memory usage'
            })
            results['potential_savings_mb'] += memory_usage * 0.5  # Rough estimate
        
        layer_idx += 1
    
    return results


# Predefined configurations for common use cases

def get_transformer_config() -> CheckpointConfig:
    """Get checkpointing config optimized for Transformer models."""
    return CheckpointConfig(
        policy=CheckpointPolicy.ATTENTION_ONLY,
        preserve_rng_state=True,
        use_reentrant=True,
        attention_layers={nn.MultiheadAttention}
    )


def get_cnn_config() -> CheckpointConfig:
    """Get checkpointing config optimized for CNN models."""
    return CheckpointConfig(
        policy=CheckpointPolicy.EVERY_N,
        interval=3,  # Checkpoint every 3 layers
        preserve_rng_state=True,
        use_reentrant=True
    )


def get_memory_optimized_config(threshold_mb: float = 100.0) -> CheckpointConfig:
    """Get checkpointing config for maximum memory optimization."""
    return CheckpointConfig(
        policy=CheckpointPolicy.MEMORY_EFFICIENT,
        memory_threshold_mb=threshold_mb,
        preserve_rng_state=True,
        use_reentrant=True
    )
