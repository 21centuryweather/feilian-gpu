"""
Neural Network Module for Wind Flow Prediction
==============================================

This module provides the FeilianNet architecture - a U-Net based deep learning model
designed for wind flow prediction around buildings. It includes enhanced GPU support,
cross-platform compatibility, and advanced training features.

Key Components:
- FeilianNet: Main U-Net architecture with configurable depth and channels
- Enhanced training functions with mixed precision and checkpointing
- Cross-platform GPU support (CUDA, MPS, CPU)
- Model utilities for saving, loading, and parameter counting

Architecture Features:
- Encoder-decoder structure with skip connections
- Configurable compression levels (max_level)
- Channel multiplication for feature extraction (chan_multi)
- Support for various activation functions (ReLU, PReLU)
- BatchNorm and dropout for regularization

Typical Usage:
    >>> model = FeilianNet(chan_multi=20, max_level=6, activation=nn.ReLU(True))
    >>> model = train_network_model_with_adam(
    ...     model, x_train, y_train,
    ...     device_preference="auto",
    ...     use_mixed_precision=True
    ... )
    >>> predictions = predict_with_model(model, x_test)

Author: Feilian Development Team
Version: 1.0.0
"""

import os
from datetime import datetime
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DataParallel
from torch.utils.data import TensorDataset

from .device_manager import DeviceManager

# ============================================================================
# Utility Functions
# ============================================================================


def predict_with_model(model, x, batch_size=8, device_manager=None):
    """
    Generates predictions using the provided model on the input data.

    Args:
        model (torch.nn.Module): The neural network model to use for predictions.
        x (numpy.ndarray): The input data to predict on.
        batch_size (int, optional): The number of samples per batch to load. Default is 8.
        device_manager (DeviceManager, optional): Device manager for handling device operations.

    Returns:
        numpy.ndarray: The predicted output corresponding to the input data.
    """
    if device_manager is None:
        from .device_manager import get_device_manager

        device_manager = get_device_manager()

    device = device_manager.device
    model = model.to(device)
    model.eval()

    y = np.empty(np.shape(x))

    # Create dataset and dataloader
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32))
    x_loader = device_manager.create_dataloader(dataset, batch_size, shuffle=False)

    idx = 0
    with torch.no_grad():
        for (xi,) in x_loader:
            xi = xi.to(device)
            yi = model(xi).cpu().detach()
            n = np.shape(yi)[0]
            y[idx : (idx + n), :, :, :] = yi.numpy()
            idx += n

    return y


def train_network_model_with_adam(
    model,
    x_train,
    y_train,
    batch_size=8,
    lr=1e-3,
    criterion=nn.L1Loss(),
    num_epochs=10,
    model_dir=".output/models",
    device_preference="auto",
    use_mixed_precision=True,
    save_checkpoints=True,
    checkpoint_interval=50,
):
    """
    Trains a neural network model using the Adam optimizer with enhanced GPU support.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x_train (torch.Tensor or numpy.ndarray): The input training data.
        y_train (torch.Tensor or numpy.ndarray): The target training data.
        batch_size (int, optional): The number of samples per batch. Default is 8.
        lr (float, optional): The learning rate for the Adam optimizer. Default is 1e-3.
        criterion (torch.nn.Module, optional): The loss function to be used. Default is nn.L1Loss().
        num_epochs (int, optional): The number of epochs to train the model. Default is 1000.
        model_dir (str, optional): The directory where the trained model will be saved. Default is ".output/models".
        device_preference (str, optional): Device preference: "auto", "mps", "cuda", or "cpu". Default is "auto".
        use_mixed_precision (bool, optional): Whether to use mixed precision training. Default is True.
        save_checkpoints (bool, optional): Whether to save periodic checkpoints. Default is True.
        checkpoint_interval (int, optional): Interval for saving checkpoints. Default is 50.

    Returns:
        torch.nn.Module: The trained neural network model.
    """
    # Initialize device manager
    device_manager = DeviceManager(device_preference)
    device = device_manager.device
    device_manager.print_device_info()

    # Prepare data
    if isinstance(x_train, np.ndarray):
        x_train = torch.tensor(x_train, dtype=torch.float32)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.float32)

    # Create dataset and dataloader
    dataset = TensorDataset(x_train, y_train)
    train_loader = device_manager.create_dataloader(dataset, batch_size, shuffle=True)

    # Optimize model for device
    model = device_manager.optimize_model(
        model, use_compile=False
    )  # Disable compile for stability

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Setup mixed precision training for CUDA
    scaler = None
    if use_mixed_precision and device.type == "cuda":
        try:
            from torch.cuda.amp import GradScaler, autocast

            scaler = GradScaler()
            print("Using mixed precision training with CUDA AMP")
        except ImportError:
            print("Mixed precision not available, using standard training")

    model.train()
    start_time = datetime.now()
    count = 0
    best_loss = float("inf")

    # Create model directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    for epoch in range(num_epochs):
        total_loss, total_numel = 0.0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward pass with optional mixed precision
            if scaler is not None:
                with autocast():
                    loss = criterion(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            curr_numel = y.numel()
            total_loss += loss.item() * curr_numel
            total_numel += curr_numel

        avg_train_loss = total_loss / total_numel
        time_elapsed = str(datetime.now() - start_time)[:-3]
        print(
            f"[{time_elapsed}] Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_train_loss:.5f}"
        )

        # Early stopping for diverging loss
        if avg_train_loss > 1e4:
            count += 1
            if count > 20:
                print("Loss is too large, terminating...")
                break
        else:
            count = 0

        # Save checkpoint
        if save_checkpoints and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"{model_dir}/checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(
                model, optimizer, epoch + 1, avg_train_loss, checkpoint_path
            )

        # Save best model
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            best_model_path = f"{model_dir}/best_model.pth"
            save_model_state(model, best_model_path)

        # Clear cache periodically
        if (epoch + 1) % 100 == 0:
            device_manager.clear_cache()

    # Save final model
    curr_time = datetime.now().strftime("%Y%m%dT%H:%M:%S")
    final_model_path = f"{model_dir}/feilian_net_{curr_time}.pth"
    save_model_state(model, final_model_path)
    print(f"Model saved to {final_model_path}")

    return model


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training checkpoint."""
    # Handle DataParallel models
    model_state = (
        model.module.state_dict()
        if isinstance(model, DataParallel)
        else model.state_dict()
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "timestamp": datetime.now().isoformat(),
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved: {path}")


def save_model_state(model, path):
    """Save only model state dict."""
    # Handle DataParallel models
    model_state = (
        model.module.state_dict()
        if isinstance(model, DataParallel)
        else model.state_dict()
    )
    torch.save(model_state, path)


def load_checkpoint(model, optimizer, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("loss", 0)


class FeilianNet(nn.Module):
    def __init__(
        self,
        conv_kernel_size=3,
        pool_kernel_size=2,
        max_level=4,
        chan_multi=16,
        activation=nn.PReLU(),
        data_type=torch.float32,
        seed=None,
    ):
        """
        Initializes the FeilianNet neural network.

        Args:
            conv_kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
            pool_kernel_size (int, optional): Size of the pooling kernel. Default is 2.
            max_level (int, optional): Maximum level of the network. Default is 4.
            chan_multi (int, optional): Channel multiplier for the network. Default is 16.
            activation (nn.Module, optional): Activation function to use. Default is nn.PReLU().
            data_type (torch.dtype, optional): Data type for the tensors. Default is torch.float32.
            seed (int, optional): Random seed for reproducibility. Default is None.

        Attributes:
            conv_kernel_size (int): Size of the convolutional kernel.
            pool_kernel_size (int): Size of the pooling kernel.
            max_level (int): Maximum level of the network.
            data_type (torch.dtype): Data type for the tensors.
            chan_multi (int): Channel multiplier for the network.
            activation (nn.Module): Activation function to use.
            compress_layers (nn.ModuleList): List of compression layers.
            switch_layer (SwitchLayer): Switch layer of the network.
            expand_layers (nn.ModuleList): List of expansion layers.
        """
        super(FeilianNet, self).__init__()
        self.conv_kernel_size, self.pool_kernel_size = (
            conv_kernel_size,
            pool_kernel_size,
        )
        self.max_level, self.data_type, self.chan_multi = (
            max_level,
            data_type,
            chan_multi,
        )
        self.activation = activation

        self.compress_layers = nn.ModuleList()
        self.switch_layer = None
        self.expand_layers = nn.ModuleList()

        convargs = {"dtype": data_type, "kernel_size": conv_kernel_size}
        if seed:
            torch.manual_seed(seed)
        for level in range(max_level):
            self.compress_layers.append(
                CompressionLayer(
                    level, chan_multi, pool_kernel_size, activation, convargs
                )
            )
        self.switch_layer = SwitchLayer(
            max_level, chan_multi, pool_kernel_size, activation, convargs
        )
        for level in range(max_level - 1, -1, -1):
            self.expand_layers.append(
                ExpansionLayer(
                    level, chan_multi, pool_kernel_size, activation, convargs
                )
            )

    def forward(self, x):
        """
        Perform a forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor to the network.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        xs = [None for _ in range(self.max_level + 1)]
        xs[0] = x
        for i, layer in enumerate(self.compress_layers):
            xs[i + 1] = layer(xs[i])
        y = self.switch_layer(xs[-1])
        for i, layer in enumerate(self.expand_layers):
            y = layer(xs[-i - 1], y)
        return y

    def count_trainable_parameters(self):
        """
        Count the number of trainable parameters in the neural network.

        This method iterates through the layers of the neural network and sums up
        the number of parameters that require gradients (i.e., trainable parameters).

        Returns:
            int: The total number of trainable parameters in the neural network.
        """
        nparams = 0
        for layer in self.compress_layers:
            nparams += sum(p.numel() for p in layer.parameters() if p.requires_grad)
        nparams += sum(
            p.numel() for p in self.switch_layer.parameters() if p.requires_grad
        )
        for layer in self.expand_layers:
            nparams += sum(p.numel() for p in layer.parameters() if p.requires_grad)
        return nparams


class CompressionLayer(nn.Module):
    def __init__(self, level, chan_multi, pool_kernel_size, activation, convargs):
        """
        Initializes the CompressionLayer.

        Args:
            level (int): The level of the compression layer. Determines the number of channels.
            chan_multi (int): The channel multiplier used to calculate the number of channels.
            pool_kernel_size (int or tuple): The size of the kernel to be used in the MaxPool2d layer.
            activation (callable): The activation function to be used in the convolutional layers.
            convargs (dict): Additional arguments to be passed to the convolutional layers.

        """
        super(CompressionLayer, self).__init__()
        self.layer = (
            nn.Sequential()
            if level == 0
            else nn.Sequential(nn.MaxPool2d(pool_kernel_size))
        )
        prev_chan = 1 if level == 0 else chan_multi * 2 ** (level - 1)
        curr_chan = chan_multi * 2**level
        self.layer.extend(
            _double_conv_layers(prev_chan, curr_chan, convargs, activation)
        )

    def forward(self, x):
        return self.layer(x)


class SwitchLayer(nn.Module):
    """
    Initializes the SwitchLayer.

    Args:
        max_level (int): The maximum level of the neural network.
        chan_multi (int): The channel multiplier.
        pool_kernel_size (int or tuple): The size of the kernel for the max pooling layer.
        activation (callable): The activation function to use.
        convargs (dict): Additional arguments for the convolutional layers.

    """

    def __init__(self, max_level, chan_multi, pool_kernel_size, activation, convargs):
        super(SwitchLayer, self).__init__()
        prev_chan = next_chan = chan_multi * 2 ** (max_level - 1)
        curr_chan = chan_multi * 2**max_level
        self.layer = nn.Sequential(
            nn.MaxPool2d(pool_kernel_size),
            *_double_conv_layers(prev_chan, curr_chan, convargs, activation),
            nn.ConvTranspose2d(
                curr_chan, next_chan, stride=pool_kernel_size, **convargs
            ),
        )

    def forward(self, x):
        return self.layer(x)


class ExpansionLayer(nn.Module):
    def __init__(self, level, chan_multi, pool_kernel_size, activation, convargs):
        """
        Initializes the ExpansionLayer.

        Args:
            level (int): The current level of the layer in the network.
            chan_multi (int): The channel multiplier.
            pool_kernel_size (int): The kernel size for pooling.
            activation (callable): The activation function to use.
            convargs (dict): Additional arguments for convolutional layers.

        """
        super(ExpansionLayer, self).__init__()
        curr_chan = chan_multi * 2**level
        next_chan = chan_multi * 2 ** (level - 1)
        self.layer = nn.Sequential(
            *_double_conv_layers(2 * curr_chan, curr_chan, convargs, activation)
        )
        if level == 0:
            self.layer.append(nn.Conv2d(curr_chan, 1, 1, dtype=convargs["dtype"]))
        else:
            self.layer.append(
                nn.ConvTranspose2d(
                    curr_chan, next_chan, stride=pool_kernel_size, **convargs
                )
            )

    def forward(self, x0, x1):
        """
        Forward pass for the neural network.

        Args:
            x0 (torch.Tensor): The first input tensor.
            x1 (torch.Tensor): The second input tensor.

        Returns:
            torch.Tensor: The output tensor after concatenation and passing through the layer.
        """
        dx, dy = x0.size()[-1] - x1.size()[-1], x0.size()[-2] - x1.size()[-2]
        if dx != 0 or dy != 0:
            x1 = F.pad(x1, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        x = torch.cat([x0, x1], dim=1)
        return self.layer(x)


def _double_conv_layers(prev_chan, curr_chan, convargs, activation):
    """
    Creates a sequence of two convolutional layers, each followed by a batch normalization layer and an activation function.

    Args:
        prev_chan (int): Number of input channels for the first convolutional layer.
        curr_chan (int): Number of output channels for both convolutional layers.
        convargs (dict): Additional arguments to pass to the convolutional layers.
        activation (nn.Module): Activation function to use. If it is an instance of nn.PReLU, a new nn.PReLU instance is created for each layer.

    Returns:
        list: A list containing the layers in the following order:
            - Conv2d layer
            - BatchNorm2d layer
            - Activation function
            - Conv2d layer
            - BatchNorm2d layer
            - Activation function
    """

    def act():
        if isinstance(activation, nn.PReLU):
            return nn.PReLU(curr_chan)
        return activation

    return [
        nn.Conv2d(prev_chan, curr_chan, padding="same", bias=False, **convargs),
        nn.BatchNorm2d(curr_chan),
        act(),
        nn.Conv2d(curr_chan, curr_chan, padding="same", bias=False, **convargs),
        nn.BatchNorm2d(curr_chan),
        act(),
    ]


# Legacy function for backward compatibility
def _init_data_loader_and_model_and_device(
    model, x_train, y_train, batch_size, device_preference="auto"
):
    """
    Legacy function - use DeviceManager directly for new code.
    Initializes the data loader, model, and device for training.

    Args:
        model (torch.nn.Module): The neural network model to be trained.
        x_train (numpy.ndarray or torch.Tensor): Training data features.
        y_train (numpy.ndarray or torch.Tensor): Training data labels.
        batch_size (int): The number of samples per batch to load.
        device_preference (str): Device preference for DeviceManager.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            - model (torch.nn.Module): The model moved to the appropriate device.
            - device (torch.device): The device (CPU or GPU) on which the model is located.
    """
    warnings.warn(
        "_init_data_loader_and_model_and_device is deprecated. Use DeviceManager directly.",
        DeprecationWarning,
        stacklevel=2,
    )

    device_manager = DeviceManager(device_preference)

    if isinstance(x_train, np.ndarray):
        x_train = torch.tensor(x_train, dtype=torch.float32)
    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(x_train, y_train)
    train_loader = device_manager.create_dataloader(dataset, batch_size, shuffle=True)
    model = device_manager.optimize_model(model)

    return train_loader, model, device_manager.device
