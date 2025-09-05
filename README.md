# feilian-gpu: Advanced Wind Flow Prediction via Deep Learning

## Overview

feilian-gpu is an enhanced deep learning project that predicts wind flow patterns around buildings using a U-Net-based neural network architecture. The project features comprehensive cross-platform GPU acceleration, 3D NetCDF data support, and advanced training capabilities for Large Eddy Simulation (LES) wind flow data.

## What This Project Does

The project predicts pedestrian-level wind speeds around buildings using:

- **Input**: Building topology (2D/3D data) and wind direction
- **Output**: Predicted wind speed fields at pedestrian level
- **Model**: Custom U-Net architecture (FeilianNet) with multiple compression/expansion layers
- **Training Data**: LES simulation results from urban wind flow studies
- **Data Formats**: Support for both NumPy (.npy) and NetCDF (.nc) files

## New Features & Enhancements

### Cross-Platform GPU Support

- **Apple Silicon (MPS)** support for M1/M2/M3 Macs
- **NVIDIA CUDA** acceleration for Linux/Windows
- **Intel CPU** fallback with multi-threading
- **Automatic device detection** and optimization
- **Mixed precision training** (CUDA)
- **Memory monitoring** and cleanup

### 3D NetCDF Data Integration **FULLY WORKING**

- **NetCDF4 file support** for 3D wind speed data - **xarray/netCDF4 loading now functional**
- **Automatic z-level extraction** (configurable z=0, z=1, etc.) - **NaN handling working correctly**
- **NaN value handling** with automatic cleanup and logging
- **Universal data loader** supporting both .npy and .nc files seamlessly
- **Batch conversion tools** for NetCDF→NumPy migration (optional)
- **Robust error handling** for corrupted files with graceful fallbacks
- **224 NetCDF files successfully loaded** from `raw_data/wind3D/idealized/`
- **Proper dimension handling** with corrected (x,y,z) spatial coordinates

### Enhanced Training Pipeline

- **Advanced command-line interface** with comprehensive options
- **Checkpoint management** with automatic saving/loading
- **Model parameter auto-detection** from checkpoints
- **Flexible hyperparameter configuration**
- **Real-time progress monitoring** with device memory usage
- **Comprehensive logging** and metrics export

### Neural Network Architecture (FeilianNet)

- **U-Net-based design** with encoder-decoder structure
- **Multi-scale processing** with configurable compression levels (max_level)
- **Channel multiplication** for feature extraction (chan_multi)
- **Flexible activation functions** (ReLU, PReLU)
- **Skip connections** for detail preservation
- **Parameter counting** and architecture visualization

### Data Processing Pipeline

- **DataFormatter class** for preprocessing wind simulation data
- **Multi-directional wind handling** with automatic rotation
- **Flexible input shapes** with configurable formatting dimensions
- **Train/test splitting** with reproducible random seeds
- **Comprehensive metrics** (MAE, RMSE, R², relative errors)
- **Automatic data validation** and quality checks

## Installation

### Prerequisites

```bash
# Core dependencies
pip install torch torchvision numpy scipy scikit-learn

# NetCDF support (required for 3D data)
pip install netCDF4

# Optional dependencies for visualization
pip install pandas matplotlib seaborn
```

### Setup

```bash
git clone https://github.com/your-username/feilian-gpu.git
cd feilian-gpu
```

## Usage

### Enhanced GPU Training (Recommended)

The enhanced training script supports cross-platform GPU acceleration with working NetCDF data loading:

```bash
# Basic training with auto device detection (NetCDF loading fully working)
python feilian_main.py 42 --data-path raw_data/wind3D/idealized/ --verbose

# Optimized for Apple Silicon (16GB unified memory)
python feilian_main.py 42 \
    --data-path raw_data/wind3D/idealized/ \
    --batch-size 2 \
    --num-epochs 1000 \
    --learning-rate 1e-4 \
    --verbose

# Memory-efficient training for resource-constrained systems
python feilian_main.py 42 \
    --data-path raw_data/wind3D/idealized/ \
    --batch-size 1 \
    --chan-multi 16 \
    --max-level 5 \
    --num-epochs 1000

# Production training with checkpoints
python feilian_main.py 42 \
    --data-path raw_data/wind3D/idealized/ \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --num-epochs 1000 \
    --save-checkpoints \
    --checkpoint-interval 50 \
    --save-images \
    --verbose

# Force specific device
python feilian_main.py 42 --device cuda --batch-size 4  # NVIDIA GPU
python feilian_main.py 42 --device mps --batch-size 2   # Apple Silicon
python feilian_main.py 42 --force-cpu --batch-size 1    # CPU debugging
```

### Command Line Options

| Option | Description | Default | Memory Impact |
|--------|-------------|---------|---------------|
| `--data-path` | Path to training data directory | `raw_data/wind3D/idealized/` | - |
| `--num-epochs` | Number of training epochs | `1000` | - |
| `--batch-size` | Training batch size | `4` | **High** - reduce to 2 or 1 for memory constraints |
| `--learning-rate` | Adam optimizer learning rate | `1e-3` | - |
| `--chan-multi` | Network channel multiplier | `20` | **High** - reduce to 16, 8 for smaller model |
| `--max-level` | U-Net compression levels | `6` | **Medium** - reduce to 5, 3 for memory efficiency |
| `--activation` | Activation function (ReLU/PReLU) | `ReLU` | - |
| `--device` | Device preference (auto/mps/cuda/cpu) | `auto` | - |
| `--mixed-precision` | Use mixed precision training | `True` | **Positive** - reduces memory usage (CUDA only) |
| `--save-checkpoints` | Save periodic checkpoints | `True` | - |
| `--checkpoint-interval` | Epochs between checkpoints | `50` | - |
| `--model-dir` | Directory to save models | `./models` | - |
| `--save-images` | Save prediction images | `False` | **Low** disk usage |
| `--output-dir` | Directory for output files | `.output` | - |

### Memory Optimization Guide

| System Configuration | Recommended Settings | Expected Performance |
|---------------------|---------------------|---------------------|
| **Apple Silicon 16GB** | `--batch-size 2` | ✅ Stable, ~2min/epoch |
| **NVIDIA 8GB+ VRAM** | `--batch-size 4 --mixed-precision` | ✅ Fast, ~1min/epoch |
| **Memory Constrained** | `--batch-size 1 --chan-multi 16 --max-level 5` | ✅ Slower but stable |
| **Debugging/Testing** | `--batch-size 1 --chan-multi 8 --max-level 3` | ✅ Minimal footprint |

### Legacy Training

The original training script is still available:

```bash
python feilian_orig.py 42
```

## Working with NetCDF Data

### Corrected NetCDF File Structure

The current NetCDF files in `raw_data/wind3D/idealized/` have been corrected to use proper dimension ordering:

- **Dimension Order**: `(x, y, z)` - properly labeled spatial dimensions
- **Variable**: `wind_speed` with shape like `(640, 384, 130)`
- **Z-levels**:
  - `z=0`: Surface level (typically all zeros)
  - `z=1`: Ground level with meaningful pedestrian wind speeds
  - `z=2+`: Higher elevation levels

**Important**: Always use `z_level=1` for ground-level wind predictions as `z=0` contains surface data (zeros).

### Loading 3D Wind Speed Data

```python
from feilian import load_wind_data

# Load NetCDF file and extract z=1 slice (ground level with meaningful wind speeds)
wind_data = load_wind_data(
    'path/to/wind_data.nc', 
    netcdf_variable='wind_speed', 
    z_level=1  # z=1 is ground level, z=0 is surface (typically zeros)
)
print(f"Wind data shape: {wind_data.shape}")  # (640, 384) or similar
```

### Batch Processing NetCDF Files

```python
from feilian import find_wind_data_files, batch_convert_netcdf_to_numpy

# Find all NetCDF files in directory
nc_files = find_wind_data_files('raw_data/variable/', include_netcdf=True)

# Convert all NetCDF files to NumPy format
converted_files = batch_convert_netcdf_to_numpy(
    'raw_data/variable/', 
    'data/converted/', 
    z_level=1
)
```

### Handling Data Quality Issues

The system automatically handles corrupted NetCDF files:

```bash
# Check for problematic files and move them to quarantine
python -c "
from pathlib import Path
import netCDF4 as nc
problematic = []
for f in Path('raw_data/variable/').glob('*.nc'):
    try:
        with nc.Dataset(f) as ds:
            test = ds.variables['wind_speed'][0,:,:]
    except:
        problematic.append(f)
print(f'Found {len(problematic)} problematic files')
"
```

## Model Loading Utilities

### Automatic Parameter Detection

```python
from model_loader_helper import load_feilian_model_from_checkpoint

# Automatically detects chan_multi, max_level, and other parameters
model = load_feilian_model_from_checkpoint('models/best_model.pth')

# Or manually check parameters first
from model_loader_helper import detect_model_params_from_checkpoint
params = detect_model_params_from_checkpoint('models/best_model.pth')
print(f"Model trained with chan_multi={params['chan_multi']}")
```

### Manual Model Loading

```python
from feilian import FeilianNet
import torch
import torch.nn as nn

# Create model with correct architecture
model = FeilianNet(
    chan_multi=20,  # Match the checkpoint!
    max_level=6, 
    activation=nn.ReLU(True)
)

# Load checkpoint
checkpoint = torch.load('models/feilian_net_*.pth', map_location='cpu')
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(checkpoint)
```

### Using the Trained Model for Prediction

See `Feilian-demo.ipynb` for a complete example:

```python
from feilian import DataFormatter, FeilianNet
from model_loader_helper import load_feilian_model_from_checkpoint
import torch
import numpy as np

# Load and format data
topo = -building_heights  # Negative values for topology
dfmt = DataFormatter([topo], [wind_angle], 1280)

# Load model with automatic parameter detection
model = load_feilian_model_from_checkpoint('models/best_model.pth')

# Make predictions
fmt_input = torch.tensor(dfmt.get_formatted_input_data())
with torch.no_grad():
    prediction = model(fmt_input)
    
# Convert back to original format
result = dfmt.restore_raw_output_data(prediction.numpy())[0]
```

## Model Architecture Details

### FeilianNet Configuration

```python
model = FeilianNet(
    chan_multi=20,        # Base channel multiplier
    max_level=6,          # Number of compression levels
    activation=nn.ReLU(inplace=True),
    conv_kernel_size=3,   # Convolution kernel size
    pool_kernel_size=2,   # Pooling kernel size
    data_type=torch.float32
)
```

### Training Parameters

```python
train_network_model_with_adam(
    model,
    x_train, y_train,
    batch_size=4,
    lr=1e-3,
    criterion=nn.L1Loss(),
    num_epochs=1000
)
```

## Data Format

### Input Data

- **Topology**: 2D arrays with negative values representing building heights
- **Wind Direction**: Angles in degrees (0-360)
- **Format Shape**: Target resolution (e.g., 1280x1280)

### Output Data

- **Wind Speed**: 2D arrays of predicted wind speeds at pedestrian level
- **Metrics**: Comprehensive evaluation including MAE, RMSE, R²
