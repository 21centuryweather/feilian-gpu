# Feilian GPU Module Documentation

This document provides a comprehensive overview of all Python files in the `feilian/` module, detailing their purpose, key classes, functions, and usage patterns.

## Table of Contents

1. [Core Module Structure](#core-module-structure)
2. [Data Processing](#data-processing)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Device Management](#device-management)
5. [Data Loading & NetCDF Support](#data-loading--netcdf-support)
6. [Performance Optimization](#performance-optimization)
7. [Memory Management](#memory-management)
8. [Mixed Precision Training](#mixed-precision-training)
9. [Gradient Checkpointing](#gradient-checkpointing)
10. [Benchmarking](#benchmarking)
11. [Legacy Components](#legacy-components)

---

## Core Module Structure

### `__init__.py`

**Purpose**: Main module initialization and public API definition

**Key Features**:

- Imports all core functionality from submodules
- Provides convenience imports for common usage patterns
- Handles optional dependencies (NetCDF support)
- Defines the public API with `__all__` exports

**Key Imports**:

```python
# Core modules
from .data_formatter import *
from .neural_network import *
from .device_manager import DeviceManager, get_device_manager

# NetCDF data loading (optional)
from .netcdf_loader import load_wind_data, convert_netcdf_to_numpy

# Performance optimization modules
from .benchmark import BenchmarkRunner, ModelFactory
from .memory_manager import MemoryManager, allocate, free
from .amp import AMPManager, autocast, scale_loss
from .checkpoint import wrap_with_checkpointing, CheckpointConfig
```

**Usage Pattern**:

```python
import feilian
device_manager = feilian.get_device_manager()
model = feilian.FeilianNet(chan_multi=20, max_level=6)
```

---

## Data Processing

### `data_formatter.py`

**Purpose**: Core data preprocessing and formatting for wind flow prediction

**Key Classes**:

- **`DataFormatter`**: Main class for processing wind simulation data with rotation and formatting

**Key Functions**:

- `_find_formatted_shape()`: Determines target formatting dimensions
- `_determine_shape_before_rotation()`: Calculates shape needed before rotation
- `_compute_expanded_shape_and_rotated_blocks()`: Handles data expansion for tiling
- `_init_all_metrics()`: Initialize comprehensive evaluation metrics
- `_compute_then_add_metrics()`: Calculate MAE, RMSE, R², and additional metrics

**DataFormatter Features**:

```python
# Initialize with raw data and wind angles
formatter = DataFormatter(raw_data, wind_angles, formatted_shape=1280)

# Get processed data
x_input = formatter.get_formatted_input_data()
y_output = formatter.get_formatted_output_data()

# Split into train/test
x_train, y_train, train_idx, x_test, y_test, test_idx = formatter.split_train_test_data(0.8, seed=42)

# Restore original format from network output
restored = formatter.restore_raw_output_data(predictions)

# Comprehensive metrics evaluation
metrics = formatter.compute_all_metrics(predictions)
# Returns: MAE, RMSE, R², relative errors, mean/std differences
```

**Data Processing Pipeline**:

1. **Input Processing**: Handles building topology (negative values) and wind directions
2. **Rotation**: Rotates data based on wind angles (90-degree increments)
3. **Expansion**: Pads and tiles data to fit target resolution
4. **Formatting**: Creates consistent input/output format for neural network
5. **Restoration**: Converts network output back to original coordinate system

---

## Neural Network Architecture

### `neural_network.py`

**Purpose**: U-Net based neural network architecture (FeilianNet) with enhanced GPU support

**Key Classes**:

#### **`FeilianNet`**: Main neural network architecture

- **U-Net Design**: Encoder-decoder with skip connections
- **Configurable Depth**: `max_level` parameter controls compression levels
- **Channel Scaling**: `chan_multi` multiplies base channel count
- **Flexible Activation**: Supports ReLU, PReLU, and custom activations

```python
model = FeilianNet(
    conv_kernel_size=3,     # Convolution kernel size
    pool_kernel_size=2,     # Pooling kernel size  
    max_level=6,           # U-Net depth levels
    chan_multi=20,         # Channel multiplier
    activation=nn.ReLU(True),  # Activation function
    data_type=torch.float32,   # Data precision
    seed=42                # Reproducibility
)
```

#### **`CompressionLayer`**: Encoder blocks

- Downsampling with MaxPool2d
- Double convolution blocks with BatchNorm
- Channel progression: `chan_multi * 2^level`

#### **`SwitchLayer`**: Bottleneck layer

- Bridge between encoder and decoder

#### **`ExpansionLayer`**: Decoder blocks  

- Upsampling with ConvTranspose2d
- Skip connection concatenation
- Progressive channel reduction

**Key Functions**:

#### **`train_network_model_with_adam()`**: Enhanced training function

```python
model = train_network_model_with_adam(
    model=model,
    x_train=x_train, 
    y_train=y_train,
    batch_size=4,
    lr=1e-3,
    criterion=nn.L1Loss(),
    num_epochs=1000,
    device_preference="auto",      # Auto-select best device
    use_mixed_precision=True,      # AMP for CUDA
    save_checkpoints=True,         # Periodic saves
    checkpoint_interval=50         # Save every 50 epochs
)
```

**Features**:

- **Cross-platform GPU support**: CUDA, MPS (Apple Silicon), CPU
- **Automatic mixed precision**: For CUDA devices
- **Multi-GPU support**: DataParallel for multiple GPUs
- **Checkpoint management**: Automatic saving and loading
- **Early stopping**: Prevents diverging loss
- **Memory optimization**: Periodic cache clearing

#### **`predict_with_model()`**: Inference function

```python
predictions = predict_with_model(
    model=trained_model,
    x=test_data,
    batch_size=8,
    device_manager=device_manager
)
```

**Model Utilities**:

```python
# Count parameters
param_count = model.count_trainable_parameters()

# Save/load models
save_model_state(model, "model.pth")
save_checkpoint(model, optimizer, epoch, loss, "checkpoint.pth")
epoch, loss = load_checkpoint(model, optimizer, "checkpoint.pth")
```

---

## Device Management

### `device_manager.py`

**Purpose**: Cross-platform GPU device management and optimization

**Key Class**: **`DeviceManager`**

**Device Support**:

- **Apple Silicon MPS**: M1/M2/M3 Mac GPU acceleration
- **NVIDIA CUDA**: Full CUDA support with multi-GPU
- **CPU Fallback**: Optimized CPU usage with threading

**Features**:

```python
# Initialize device manager
dm = DeviceManager(device_preference="auto")  # auto, mps, cuda, cpu
dm = DeviceManager("cuda", force_cpu=False)

# Get device information
info = dm.get_device_info()
print(f"Device: {info['device']}")
print(f"GPU Name: {info.get('gpu_name', 'N/A')}")
print(f"Memory: {info.get('memory_total', 'N/A')}")

# Optimize model for device
model = dm.optimize_model(model, use_compile=True)

# Create optimized DataLoader
dataloader = dm.create_dataloader(
    dataset, 
    batch_size=32, 
    shuffle=True, 
    num_workers=None  # Auto-select optimal workers
)

# Memory management
dm.clear_cache()  # Clear device cache
tensor = dm.move_to_device(tensor)
```

**Device Selection Logic**:

1. **Auto Mode**: MPS > CUDA > CPU (by performance)
2. **Availability Checks**: Validates device capabilities
3. **Optimization**: Device-specific model optimization
4. **Worker Selection**: Optimal DataLoader configuration

**Convenience Functions**:

```python
# Factory functions
device_manager = get_device_manager("auto")
best_device = get_best_device("auto")
print_available_devices()  # System information
```

---

## Data Loading & NetCDF Support

### `netcdf_loader.py`

**Purpose**: Comprehensive NetCDF file support for 3D wind data - **xarray/netCDF4 integration working**

**Key Features** - **All Tested and Working**:

- **3D to 2D Extraction**: Extracts specific z-levels from 3D wind data ✅
- **NaN Handling**: Automatic detection and replacement of NaN values ✅
- **Format Detection**: Auto-detects file types (.nc, .npy) ✅
- **Error Handling**: Robust handling of corrupted files ✅
- **Batch Processing**: Convert multiple files efficiently ✅
- **224 NetCDF Files**: Successfully loaded from `raw_data/wind3D/idealized/` ✅

**Core Functions**:

#### **`load_netcdf_wind_speed()`**: Load 3D NetCDF data

```python
wind_data = load_netcdf_wind_speed(
    filepath="wind_data.nc",
    variable_name="wind_speed",  # Variable to extract
    z_level=1,                   # Z-level (0=surface, 1=ground)
    transpose=False              # Optional transpose
)
# Returns: 2D numpy array (130, 130)
```

#### **`load_wind_data()`**: Universal data loader

```python
# Works with both .nc and .npy files
data = load_wind_data(
    filepath="data.nc",          # .nc or .npy file
    netcdf_variable="wind_speed",
    z_level=1,
    transpose=False
)
```

#### **`find_wind_data_files()`**: Directory scanning

```python
files = find_wind_data_files(
    data_path="raw_data/",
    include_netcdf=True,     # Include .nc files
    include_numpy=True       # Include .npy files
)
```

#### **Batch Processing**

```python
# Convert all NetCDF files to numpy format
converted = batch_convert_netcdf_to_numpy(
    netcdf_dir="raw_data/netcdf/",
    output_dir="processed/numpy/",
    z_level=1
)
```

**Data Quality Features**:

- **NaN Detection**: Identifies and replaces NaN values with zeros
- **Corruption Handling**: Graceful handling of unreadable files
- **Variable Detection**: Attempts multiple common variable names
- **Dimension Analysis**: Automatically determines data structure

**Filename Parsing**:

```python
# Extract wind angle from filename patterns
angle = parse_wind_angle_from_netcdf_filename("case_deg180.nc")  # Returns 180
angle = parse_wind_angle_from_netcdf_filename("case_270deg.nc")  # Returns 270
```

---

## Performance Optimization

### `benchmark.py`

**Purpose**: Comprehensive benchmarking suite for performance analysis

**Key Classes**:

#### **`BenchmarkRunner`**: Main benchmarking engine

```python
runner = BenchmarkRunner(device_manager)

# Single model benchmark
result = runner.benchmark_model(
    model_name="cnn",           # cnn, transformer, diffusion
    batch_size=32,
    sequence_length=128,
    num_iterations=50,
    device_preference="auto",
    warmup_iterations=10
)

# Comprehensive benchmark across configurations  
results = runner.run_comprehensive_benchmark(
    models=['cnn', 'transformer'],
    batch_sizes=[16, 32, 64],
    devices=['auto', 'cuda', 'cpu']
)

# Save results
runner.save_results("benchmark_output/")
```

#### **`BenchmarkResult`**: Structured performance metrics

```python
@dataclass
class BenchmarkResult:
    model_name: str
    device: str
    batch_size: int
    
    # Timing metrics
    forward_time_ms: float
    backward_time_ms: float
    total_time_ms: float
    throughput_samples_per_sec: float
    
    # Memory metrics
    peak_memory_mb: float
    memory_allocated_mb: float
    memory_reserved_mb: float
    
    # System metrics
    cpu_percent: float
    system_memory_mb: float
    
    # Metadata
    timestamp: str
    torch_version: str
    device_name: str
```

#### **`ModelFactory`**: Standard benchmark models

```python
# Create benchmark models
cnn_model = ModelFactory.create_cnn(input_channels=3, num_classes=1000)
transformer_model = ModelFactory.create_transformer(vocab_size=32000, d_model=512)
diffusion_model = ModelFactory.create_diffusion_unet(in_channels=3, out_channels=3)
```

**CLI Interface**:

```bash
python -m feilian.benchmark --models cnn transformer --batch-sizes 16 32 --devices auto cuda
```

**Output Formats**:

- **JSON**: Structured data for programmatic analysis
- **CSV**: Spreadsheet-compatible format
- **Console**: Real-time progress and summary

---

## Memory Management

### `memory_manager.py`

**Purpose**: Advanced cross-platform memory management and optimization

**Key Classes**:

#### **`MemoryManager`**: Central memory coordinator

```python
memory_mgr = MemoryManager(device_manager)

# Unified tensor allocation
tensor = memory_mgr.allocate(
    shape=(1024, 1024),
    dtype=torch.float32,
    device="auto",              # Device selection
    requires_grad=False
)

# Explicit memory freeing
memory_mgr.free(tensor)

# Memory optimization
memory_mgr.optimize_memory("cuda")  # Specific device
memory_mgr.optimize_memory()        # All devices

# Memory statistics
stats = memory_mgr.get_memory_stats("cuda")
print(f"Allocated: {stats.allocated_mb:.1f} MB")
print(f"Peak: {stats.peak_allocated_mb:.1f} MB")
print(f"Fragmentation: {stats.fragmentation_ratio:.2%}")
```

#### **Device-Specific Allocators**

**`CUDAMemoryAllocator`**: CUDA optimization

- **Stream Management**: Multiple CUDA streams for async operations
- **Cache Management**: Efficient memory reuse
- **Multi-GPU Support**: Per-device memory tracking

**`MPSMemoryAllocator`**: Apple Silicon optimization  

- **MPS Cache**: Specialized caching for Metal Performance Shaders
- **Memory Tracking**: Limited but functional memory monitoring

**`CPUMemoryAllocator`**: CPU optimization

- **Memory Pooling**: Reuse tensors of similar shapes
- **Process Memory**: Integration with system memory monitoring

#### **`TensorLifecycleTracker`**: Memory lifecycle management

```python
# Automatic tracking of tensor allocations
tracker = TensorLifecycleTracker()
tracker.register_allocation(tensor)
tracker.update_access(tensor)

# Identify stale tensors
stale_tensors = tracker.get_stale_tensors(max_age_seconds=300)
```

#### **`MemoryStats`**: Comprehensive memory metrics

```python
@dataclass
class MemoryStats:
    allocated_bytes: int
    reserved_bytes: int
    free_bytes: int
    total_allocations: int
    active_allocations: int
    peak_allocated_bytes: int
    fragmentation_ratio: float
    
    # Convenience properties
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
```

**Background Optimization**:

- **Automatic Cleanup**: Background thread for stale memory cleanup
- **Periodic Optimization**: Regular memory defragmentation
- **Lifecycle Management**: Weak references for automatic cleanup

**Convenience Functions**:

```python
# Global memory manager functions
tensor = feilian.memory.allocate((512, 512), device="mps")
feilian.memory.free(tensor)
stats = feilian.memory.get_memory_stats()
feilian.memory.optimize_memory()
```

---

## Mixed Precision Training

### `amp.py`

**Purpose**: Cross-platform Automatic Mixed Precision (AMP) with unified API

**Key Classes**:

#### **`AMPManager`**: Central AMP coordination

```python
amp_manager = AMPManager(device_manager, config)

# Auto-cast context
with amp_manager.autocast(device="auto", dtype=torch.float16):
    output = model(input)
    loss = criterion(output, target)

# Scale loss and step optimizer
scaled_loss = amp_manager.scale_loss(loss)
success = amp_manager.backward_and_step(loss, optimizer, max_norm=1.0)
```

#### **`GradScaler`**: Cross-platform gradient scaling

```python
scaler = GradScaler(device, init_scale=2**16)

# Scale gradients
scaled_loss = scaler.scale(loss)
scaled_loss.backward()

# Step with overflow detection
scaler.unscale_(optimizer)
success = scaler.step(optimizer)
scaler.update()
```

#### **`AutocastContext`**: Device-specific autocast

```python
# CUDA autocast
with AutocastContext(cuda_device, torch.float16):
    output = model(input)

# CPU/MPS fallback autocast
with AutocastContext(cpu_device, torch.bfloat16):
    output = model(input)
```

#### **`AMPConfig`**: Configuration management

```python
@dataclass
class AMPConfig:
    enabled: bool = True
    precision_mode: PrecisionMode = PrecisionMode.FP16
    init_scale: float = 2.0**16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    enable_caching: bool = True
```

**Platform Support**:

- **CUDA**: Native `torch.cuda.amp` integration
- **CPU**: Custom autocast with operation patching
- **MPS**: Limited mixed precision with fallbacks

**Convenience Functions**:

```python
# Global AMP functions
with feilian.amp.autocast("cuda", torch.float16):
    output = model(input)

scaled_loss = feilian.amp.scale_loss(loss, "cuda")
success = feilian.amp.backward_and_step(loss, optimizer, max_norm=1.0)
```

**Features**:

- **Smart Fallbacks**: Graceful degradation on unsupported platforms
- **Automatic Scaling**: Dynamic loss scaling with overflow detection
- **Memory Efficiency**: Reduced memory usage with mixed precision
- **Performance Gains**: Faster training on modern GPUs

---

## Gradient Checkpointing

### `checkpoint.py`

**Purpose**: Flexible gradient checkpointing for memory optimization

**Key Classes**:

#### **`CheckpointConfig`**: Checkpointing configuration

```python
@dataclass
class CheckpointConfig:
    policy: CheckpointPolicy = CheckpointPolicy.EVERY_N
    interval: int = 2
    memory_threshold_mb: float = 100.0
    preserve_rng_state: bool = True
    use_reentrant: bool = True
```

#### **Checkpointing Policies**

**`EveryNPolicy`**: Regular interval checkpointing

```python
policy = EveryNPolicy(interval=3)  # Checkpoint every 3 layers
```

**`AttentionOnlyPolicy`**: Attention layer targeting

```python
policy = AttentionOnlyPolicy(attention_types={nn.MultiheadAttention})
```

**`HeavyComputePolicy`**: Computational complexity based

```python
policy = HeavyComputePolicy(heavy_types={
    nn.Conv2d, nn.Conv3d, nn.Linear, nn.LSTM
})
```

**`MemoryEfficientPolicy`**: Memory usage based

```python
policy = MemoryEfficientPolicy(memory_threshold_mb=100.0)
```

**`CustomPolicy`**: User-defined logic

```python
def custom_should_checkpoint(layer, idx, memory_usage):
    return isinstance(layer, nn.Conv2d) and idx % 2 == 0

policy = CustomPolicy(custom_should_checkpoint)
```

#### **Model Wrapping**

```python
# Wrap entire model
checkpointed_model = wrap_with_checkpointing(
    model, 
    config=CheckpointConfig(policy=CheckpointPolicy.EVERY_N, interval=2),
    recursive=True
)

# Predefined configurations
transformer_config = get_transformer_config()
cnn_config = get_cnn_config()
memory_config = get_memory_optimized_config(threshold_mb=50.0)
```

#### **Analysis Tools**

```python
# Analyze checkpointing opportunities
analysis = analyze_checkpointing_opportunities(model, memory_threshold_mb=50.0)
print(f"Total layers: {analysis['total_layers']}")
print(f"Recommended checkpoints: {len(analysis['recommended_checkpoints'])}")
print(f"Potential savings: {analysis['potential_savings_mb']:.1f} MB")

# Memory estimation
memory_usage = estimate_memory_usage(layer)  # Returns MB
```

#### **Sequential Checkpointing**

```python
# Checkpoint sequential operations
sequential_fn = checkpoint_sequential(
    functions=[layer1, layer2, layer3, layer4],
    segments=2,  # Divide into 2 checkpointed segments
    preserve_rng_state=True
)

result = sequential_fn(input_tensor)
```

#### **Function Decorators**

```python
@checkpoint_function(preserve_rng_state=True)
def heavy_computation(x):
    return expensive_operation(x)
```

---

## Benchmarking

### `benchmark.py` (Detailed Features)

**Advanced Benchmarking Capabilities**:

#### **Memory Profiling**

```python
# Detailed memory statistics
def _get_memory_stats(device):
    if device.type == 'cuda':
        return {
            'allocated_mb': torch.cuda.memory_allocated(device) / 1024**2,
            'reserved_mb': torch.cuda.memory_reserved(device) / 1024**2,
            'peak_memory_mb': torch.cuda.max_memory_allocated(device) / 1024**2
        }
```

#### **System Integration**

```python
# System resource monitoring
cpu_percent = psutil.cpu_percent()
system_memory = psutil.virtual_memory().used / 1024**2
```

#### **Synchronization Handling**

```python
# Device-specific synchronization
if device.type == 'cuda':
    torch.cuda.synchronize(device)
elif device.type == 'mps':
    torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
```

#### **Statistical Analysis**

```python
# Exclude warmup iterations from statistics
stable_start = max(1, num_iterations // 10)
avg_time = sum(times[stable_start:]) / len(times[stable_start:])
throughput = batch_size / avg_time
```

---

## Legacy Components

### `neural_network_backup.py`

**Purpose**: Original neural network implementation for compatibility

**Key Differences from Enhanced Version**:

- **Simple Device Selection**: Basic CUDA/CPU detection
- **No Mixed Precision**: Standard FP32 training only
- **Basic DataParallel**: Simple multi-GPU support
- **Limited Checkpointing**: No advanced checkpoint management
- **No Device Manager Integration**: Direct PyTorch device handling

**Usage**: Maintained for backward compatibility with existing scripts

```python
# Legacy training function
model = train_network_model_with_adam(
    model, x_train, y_train, 
    batch_size=8, lr=1e-3,
    criterion=nn.L1Loss(), 
    num_epochs=1000,
    model_dir=".output/models"
)

# Legacy device initialization
train_loader, model, device = _init_data_loader_and_model_and_device(
    model, x_train, y_train, batch_size
)
```

---

## Usage Examples

### Basic Training Pipeline

```python
import feilian
import torch.nn as nn

# 1. Load and format data
formatter = feilian.DataFormatter(raw_data, wind_angles, 1280)
x_train, y_train, _, x_test, y_test, _ = formatter.split_train_test_data(0.8)

# 2. Create model with device management
device_manager = feilian.get_device_manager("auto")
model = feilian.FeilianNet(chan_multi=20, max_level=6, activation=nn.ReLU(True))

# 3. Train with advanced features
model = feilian.train_network_model_with_adam(
    model=model,
    x_train=x_train,
    y_train=y_train,
    batch_size=4,
    lr=1e-3,
    num_epochs=1000,
    device_preference="auto",
    use_mixed_precision=True,
    save_checkpoints=True
)

# 4. Evaluate
predictions = feilian.predict_with_model(model, x_test, batch_size=8)
metrics = formatter.compute_all_metrics(predictions)
print(f"Test MAE: {np.mean(metrics['mae']):.4f}")
```

### Advanced Memory Management

```python
import feilian

# Configure memory management
memory_manager = feilian.get_memory_manager()
memory_manager.set_memory_pool_size(512, device="cuda")  # 512 MB pool

# Allocate tensors efficiently
large_tensor = feilian.memory.allocate((2048, 2048), device="auto")

# Monitor memory usage
stats = feilian.memory.get_memory_stats("cuda")
print(f"Memory usage: {stats.allocated_mb:.1f}/{stats.reserved_mb:.1f} MB")

# Optimize memory periodically
feilian.memory.optimize_memory()
```

### Comprehensive Benchmarking

```python
import feilian

# Run benchmarks across all configurations
runner = feilian.BenchmarkRunner()
results = runner.run_comprehensive_benchmark(
    models=['cnn', 'transformer'],
    batch_sizes=[16, 32, 64],
    devices=['auto', 'cuda', 'mps', 'cpu']
)

# Analyze results
best_perf = max(results, key=lambda r: r.throughput_samples_per_sec)
print(f"Best: {best_perf.model_name} on {best_perf.device}")
print(f"Throughput: {best_perf.throughput_samples_per_sec:.2f} samples/sec")

# Save comprehensive report
runner.save_results("benchmark_results/")
```

---

## Integration Guidelines

### Device Management Integration

All modules integrate with the `DeviceManager` for consistent cross-platform support:

```python
# Consistent device handling across modules
device_manager = feilian.get_device_manager("auto")
memory_manager = feilian.MemoryManager(device_manager)
amp_manager = feilian.AMPManager(device_manager)
benchmark_runner = feilian.BenchmarkRunner(device_manager)
```

### Error Handling

Robust error handling with graceful fallbacks:

```python
try:
    # Attempt advanced feature
    with feilian.amp.autocast("cuda"):
        output = model(input)
except Exception as e:
    # Fallback to standard precision
    logger.warning(f"AMP failed: {e}, using FP32")
    output = model(input)
```

### Configuration Management

Centralized configuration for reproducible results:

```python
# Training configuration
config = {
    'device': 'auto',
    'mixed_precision': True,
    'checkpointing': True,
    'memory_optimization': True
}

# Apply configuration across modules
device_manager = feilian.DeviceManager(config['device'])
amp_config = feilian.AMPConfig(enabled=config['mixed_precision'])
```

This comprehensive module provides a complete ecosystem for wind flow prediction with advanced GPU acceleration, memory management, and performance optimization across all major platforms.
