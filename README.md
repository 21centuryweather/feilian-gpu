# feilian-gpu: Wind Flow Prediction via Deep Learning

## Overview

feilian-gpu is a deep learning project that predicts wind flow patterns around buildings using a U-Net-based neural network architecture. The project leverages GPU acceleration for training and inference on Large Eddy Simulation (LES) wind flow data.

## What This Project Does

The project aims to predict pedestrian-level wind speeds around buildings using:

- **Input**: Building topology (2D height maps) and wind direction
- **Output**: Predicted wind speed fields at pedestrian level
- **Model**: Custom U-Net architecture (FeilianNet) with multiple compression/expansion layers
- **Training Data**: LES simulation results from urban wind flow studies

## Key Features

### Neural Network Architecture (FeilianNet)
- **U-Net-based design** with encoder-decoder structure
- **Multi-scale processing** with configurable compression levels (max_level)
- **Channel multiplication** for feature extraction (chan_multi)
- **Flexible activation functions** (ReLU, PReLU)
- **Skip connections** for detail preservation

### Data Processing Pipeline
- **DataFormatter class** for preprocessing wind simulation data
- **Multi-directional wind handling** with automatic rotation
- **Flexible input shapes** with configurable formatting dimensions
- **Train/test splitting** with reproducible random seeds
- **Comprehensive metrics** (MAE, RMSE, R², relative errors)

### GPU Acceleration
- **PyTorch DataParallel** support for multi-GPU training
- **Automatic device detection** (CUDA/CPU)
- **Efficient batch processing** with configurable batch sizes
- **Memory optimization** for large-scale simulations

## Installation

### Prerequisites
- Python 3.7+
- PyTorch 1.8+
- NumPy
- SciPy
- scikit-learn
- netCDF4
- pandas
- matplotlib

### Setup
```bash
git clone https://github.com/your-username/feilian-gpu.git
```

## Usage

### Training a Model

```bash
python feilian_orig.py <seed>
```

Example:
```bash
python feilian_orig.py 42
```

### Using the Trained Model

See `Feilian-demo.ipynb` for a complete example:

```python
from feilian import DataFormatter, FeilianNet
import torch
import numpy as np

# Load and format data
topo = -building_heights  # Negative values for topology
dfmt = DataFormatter([topo], [wind_angle], 1280)

# Create and load model
model = FeilianNet(chan_multi=16, max_level=6, activation=torch.nn.ReLU(True))
model.load_state_dict(torch.load('models/feilian_net_*.pth'))

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
