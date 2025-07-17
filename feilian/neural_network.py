import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import DataParallel
from torch.utils.data import TensorDataset, DataLoader


def predict_with_model(model, x, batch_size=8):
    y = np.empty(np.shape(x))
    x_loader = DataLoader(TensorDataset(torch.tensor(x)), batch_size=batch_size, shuffle=False)
    idx = 0
    for (xi,) in x_loader:
        yi = model(xi).cpu().detach()
        n = np.shape(yi)[0]
        y[idx:(idx + n), :, :, :] = yi
        idx += n
    return y


def train_network_model_with_adam(model, x_train, y_train, batch_size=8, lr=1e-3,
                                  criterion=nn.L1Loss(), num_epochs=1000, model_dir=".output/models"):
    train_loader, model, device = _init_data_loader_and_model_and_device(model, x_train, y_train, batch_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    start_time = datetime.now()
    count = 0
    for epoch in range(num_epochs):
        total_loss, total_numel = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            curr_numel = y.numel()
            total_loss += loss.item() * curr_numel
            total_numel += curr_numel

        avg_train_loss = total_loss / total_numel
        time_elapsed = str(datetime.now() - start_time)[:-3]
        print(f"[{time_elapsed}] Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_train_loss:.5f}")
        if avg_train_loss > 1e4:
            count += 1
            if count > 20:
                print(f"Loss is too large, terminating...")
                return model

        count = 0

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    curr_time = datetime.now().strftime('%Y%m%dT%H:%M:%S')
    torch.save(model.state_dict(), f"{model_dir}/feilian_net_{curr_time}.pth")
    return model


class FeilianNet(nn.Module):
    def __init__(self, conv_kernel_size=3, pool_kernel_size=2, max_level=4, chan_multi=16,
                 activation=nn.PReLU(), data_type=torch.float32, seed=None):
        super(FeilianNet, self).__init__()
        self.conv_kernel_size, self.pool_kernel_size = conv_kernel_size, pool_kernel_size
        self.max_level, self.data_type, self.chan_multi = max_level, data_type, chan_multi
        self.activation = activation

        self.compress_layers = nn.ModuleList()
        self.switch_layer = None
        self.expand_layers = nn.ModuleList()

        convargs = {"dtype": data_type, "kernel_size": conv_kernel_size}
        if seed:
            torch.manual_seed(seed)
        for level in range(max_level):
            self.compress_layers.append(CompressionLayer(level, chan_multi, pool_kernel_size, activation, convargs))
        self.switch_layer = SwitchLayer(max_level, chan_multi, pool_kernel_size, activation, convargs)
        for level in range(max_level - 1, -1, -1):
            self.expand_layers.append(ExpansionLayer(level, chan_multi, pool_kernel_size, activation, convargs))

    def forward(self, x):
        xs = [None for _ in range(self.max_level + 1)]
        xs[0] = x
        for i, layer in enumerate(self.compress_layers):
            xs[i + 1] = layer(xs[i])
        y = self.switch_layer(xs[-1])
        for i, layer in enumerate(self.expand_layers):
            y = layer(xs[-i - 1], y)
        return y

    def count_trainable_parameters(self):
        nparams = 0
        for layer in self.compress_layers:
            nparams += sum(p.numel() for p in layer.parameters() if p.requires_grad)
        nparams += sum(p.numel() for p in self.switch_layer.parameters() if p.requires_grad)
        for layer in self.expand_layers:
            nparams += sum(p.numel() for p in layer.parameters() if p.requires_grad)
        return nparams


class CompressionLayer(nn.Module):
    def __init__(self, level, chan_multi, pool_kernel_size, activation, convargs):
        super(CompressionLayer, self).__init__()
        self.layer = nn.Sequential() if level == 0 else nn.Sequential(nn.MaxPool2d(pool_kernel_size))
        prev_chan = 1 if level == 0 else chan_multi * 2 ** (level - 1)
        curr_chan = chan_multi * 2 ** level
        self.layer.extend(_double_conv_layers(prev_chan, curr_chan, convargs, activation))

    def forward(self, x):
        return self.layer(x)


class SwitchLayer(nn.Module):
    def __init__(self, max_level, chan_multi, pool_kernel_size, activation, convargs):
        super(SwitchLayer, self).__init__()
        prev_chan = next_chan = chan_multi * 2 ** (max_level - 1)
        curr_chan = chan_multi * 2 ** max_level
        self.layer = nn.Sequential(nn.MaxPool2d(pool_kernel_size),
                                   *_double_conv_layers(prev_chan, curr_chan, convargs, activation),
                                   nn.ConvTranspose2d(curr_chan, next_chan, stride=pool_kernel_size, **convargs))

    def forward(self, x):
        return self.layer(x)


class ExpansionLayer(nn.Module):
    def __init__(self, level, chan_multi, pool_kernel_size, activation, convargs):
        super(ExpansionLayer, self).__init__()
        curr_chan = chan_multi * 2 ** level
        next_chan = chan_multi * 2 ** (level - 1)
        self.layer = nn.Sequential(*_double_conv_layers(2 * curr_chan, curr_chan, convargs, activation))
        if level == 0:
            self.layer.append(nn.Conv2d(curr_chan, 1, 1, dtype=convargs["dtype"]))
        else:
            self.layer.append(nn.ConvTranspose2d(curr_chan, next_chan, stride=pool_kernel_size, **convargs))

    def forward(self, x0, x1):
        dx, dy = x0.size()[-1] - x1.size()[-1], x0.size()[-2] - x1.size()[-2]
        if dx != 0 or dy != 0:
            x1 = F.pad(x1, (dx // 2, dx - dx // 2, dy // 2, dy - dy // 2))
        x = torch.cat([x0, x1], dim=1)
        return self.layer(x)


def _init_data_loader_and_model_and_device(model, x_train, y_train, batch_size):
    num_gpus = torch.cuda.device_count()
    dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if num_gpus > 1:
        model = DataParallel(model)
    return train_loader, model, device


def _double_conv_layers(prev_chan, curr_chan, convargs, activation):
    def act():
        if isinstance(activation, nn.PReLU):
            return nn.PReLU(curr_chan)
        return activation

    return [nn.Conv2d(prev_chan, curr_chan, padding="same", bias=False, **convargs),
            nn.BatchNorm2d(curr_chan), act(),
            nn.Conv2d(curr_chan, curr_chan, padding="same", bias=False, **convargs),
            nn.BatchNorm2d(curr_chan), act()]
