"""
Feilian GPU Benchmark Suite
===========================

A comprehensive benchmarking harness for measuring performance across
CPU, Apple Silicon MPS, and NVIDIA CUDA devices. This module provides
structured benchmarking capabilities with detailed performance metrics,
memory usage tracking, and cross-platform compatibility.

Key Features:
- Multi-device benchmarking (CPU, CUDA, MPS)
- Comprehensive performance metrics collection
- Memory usage profiling and optimization
- Statistical analysis with warmup periods
- Standard model factories for consistent testing
- Export capabilities (JSON, CSV formats)
- CLI interface for automated benchmarking

Benchmark Models:
- CNN: Convolutional neural network for image tasks
- Transformer: Encoder-only transformer for NLP tasks
- Diffusion U-Net: U-Net architecture for generative tasks

Performance Metrics:
- Timing: Forward/backward pass times, throughput
- Memory: Peak usage, allocation patterns, fragmentation
- System: CPU utilization, system memory usage
- Hardware: Device-specific performance characteristics

Typical Usage:
    >>> runner = BenchmarkRunner()
    >>> result = runner.benchmark_model("cnn", batch_size=32)
    >>> results = runner.run_comprehensive_benchmark()
    >>> runner.save_results("benchmark_output/")

Command Line:
    python -m feilian.benchmark --models cnn transformer --devices auto cuda

Author: Feilian Development Team
Version: 1.0.0
"""

import os
import json
import csv
import time
import tracemalloc
import psutil
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .device_manager import DeviceManager


@dataclass
class BenchmarkResult:
    """Structured result from a single benchmark run."""
    model_name: str
    device: str
    batch_size: int
    sequence_length: int
    
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


class ModelFactory:
    """Factory for creating standard benchmark models."""
    
    @staticmethod
    def create_cnn(input_channels: int = 3, num_classes: int = 1000) -> nn.Module:
        """Simple CNN for image classification benchmarks."""
        return nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    @staticmethod
    def create_transformer(vocab_size: int = 32000, d_model: int = 512, 
                         nhead: int = 8, num_layers: int = 6) -> nn.Module:
        """Simple Transformer for language modeling benchmarks."""
        return nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=2048,
                    dropout=0.1,
                    batch_first=True
                ),
                num_layers=num_layers
            ),
            nn.Linear(d_model, vocab_size)
        )
    
    @staticmethod
    def create_diffusion_unet(in_channels: int = 3, out_channels: int = 3) -> nn.Module:
        """Simple U-Net style model for diffusion benchmarks."""
        class SimpleUNet(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # Encoder
                self.enc1 = nn.Sequential(
                    nn.Conv2d(in_channels, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                self.pool1 = nn.MaxPool2d(2)
                
                self.enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                self.pool2 = nn.MaxPool2d(2)
                
                # Bottleneck
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True)
                )
                
                # Decoder
                self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec2 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True)
                )
                
                self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self.dec1 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                )
                
                self.final = nn.Conv2d(64, out_channels, 1)
                
            def forward(self, x):
                # Encoder
                enc1 = self.enc1(x)
                pool1 = self.pool1(enc1)
                
                enc2 = self.enc2(pool1)
                pool2 = self.pool2(enc2)
                
                # Bottleneck
                bottleneck = self.bottleneck(pool2)
                
                # Decoder
                up2 = self.upconv2(bottleneck)
                merge2 = torch.cat([up2, enc2], dim=1)
                dec2 = self.dec2(merge2)
                
                up1 = self.upconv1(dec2)
                merge1 = torch.cat([up1, enc1], dim=1)
                dec1 = self.dec1(merge1)
                
                return self.final(dec1)
        
        return SimpleUNet(in_channels, out_channels)


class BenchmarkRunner:
    """Main benchmark execution engine."""
    
    def __init__(self, device_manager: DeviceManager = None):
        self.device_manager = device_manager or DeviceManager()
        self.results: List[BenchmarkResult] = []
    
    def _get_memory_stats(self, device: torch.device) -> Dict[str, float]:
        """Get memory statistics for the given device."""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2
        elif device.type == 'mps':
            allocated = torch.mps.current_allocated_memory() / 1024**2 if hasattr(torch.mps, 'current_allocated_memory') else 0
            reserved = 0  # MPS doesn't expose reserved memory
            max_allocated = allocated
        else:
            allocated = reserved = max_allocated = 0
        
        return {
            'allocated_mb': allocated,
            'reserved_mb': reserved,
            'peak_memory_mb': max_allocated
        }
    
    def _create_sample_data(self, model_name: str, batch_size: int, 
                          sequence_length: int, device: torch.device):
        """Create sample input data for the given model."""
        if model_name == 'cnn':
            # Image data: [batch_size, channels, height, width]
            return torch.randn(batch_size, 3, 224, 224, device=device)
        elif model_name == 'transformer':
            # Sequence data: [batch_size, sequence_length]
            return torch.randint(0, 32000, (batch_size, sequence_length), device=device)
        elif model_name == 'diffusion':
            # Image data for diffusion: [batch_size, channels, height, width]
            return torch.randn(batch_size, 3, 64, 64, device=device)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def benchmark_model(self, model_name: str, batch_size: int = 32, 
                       sequence_length: int = 128, num_iterations: int = 50,
                       device_preference: str = 'auto', warmup_iterations: int = 10) -> BenchmarkResult:
        """Benchmark a single model configuration."""
        
        # Get device
        device = self.device_manager.get_device(device_preference)
        device_info = self.device_manager.get_device_info()
        
        # Create model
        if model_name == 'cnn':
            model = ModelFactory.create_cnn()
        elif model_name == 'transformer':
            model = ModelFactory.create_transformer()
        elif model_name == 'diffusion':
            model = ModelFactory.create_diffusion_unet()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        model = model.to(device)
        model.train()
        
        # Create optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss() if model_name in ['cnn', 'transformer'] else nn.MSELoss()
        
        # Create sample data
        input_data = self._create_sample_data(model_name, batch_size, sequence_length, device)
        
        if model_name == 'cnn':
            target = torch.randint(0, 1000, (batch_size,), device=device)
        elif model_name == 'transformer':
            target = torch.randint(0, 32000, (batch_size, sequence_length), device=device)
        else:  # diffusion
            target = torch.randn_like(input_data)
        
        # Clear memory stats
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        elif device.type == 'mps':
            torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
        
        # Start memory tracking
        tracemalloc.start()
        
        # Warmup
        for _ in range(warmup_iterations):
            optimizer.zero_grad()
            output = model(input_data)
            if model_name == 'transformer':
                output = output.view(-1, output.size(-1))
                target_flat = target.view(-1)
                loss = criterion(output, target_flat)
            else:
                loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elif device.type == 'mps':
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
        
        # Actual benchmark
        forward_times = []
        backward_times = []
        total_times = []
        
        for _ in range(num_iterations):
            # Forward pass timing
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elif device.type == 'mps':
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            
            start_time = time.perf_counter()
            
            optimizer.zero_grad()
            output = model(input_data)
            
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elif device.type == 'mps':
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            
            forward_time = time.perf_counter() - start_time
            forward_times.append(forward_time)
            
            # Backward pass timing
            start_time = time.perf_counter()
            
            if model_name == 'transformer':
                output = output.view(-1, output.size(-1))
                target_flat = target.view(-1)
                loss = criterion(output, target_flat)
            else:
                loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elif device.type == 'mps':
                torch.mps.synchronize() if hasattr(torch.mps, 'synchronize') else None
            
            backward_time = time.perf_counter() - start_time
            backward_times.append(backward_time)
            
            total_times.append(forward_time + backward_time)
        
        # Get final memory stats
        memory_stats = self._get_memory_stats(device)
        
        # Get system stats
        cpu_percent = psutil.cpu_percent()
        system_memory = psutil.virtual_memory().used / 1024**2
        
        # Calculate averages (exclude first few iterations for stability)
        stable_start = max(1, num_iterations // 10)
        avg_forward_time = sum(forward_times[stable_start:]) / len(forward_times[stable_start:])
        avg_backward_time = sum(backward_times[stable_start:]) / len(backward_times[stable_start:])
        avg_total_time = sum(total_times[stable_start:]) / len(total_times[stable_start:])
        
        # Calculate throughput
        throughput = batch_size / avg_total_time
        
        # Create result
        result = BenchmarkResult(
            model_name=model_name,
            device=str(device),
            batch_size=batch_size,
            sequence_length=sequence_length,
            forward_time_ms=avg_forward_time * 1000,
            backward_time_ms=avg_backward_time * 1000,
            total_time_ms=avg_total_time * 1000,
            throughput_samples_per_sec=throughput,
            peak_memory_mb=memory_stats['peak_memory_mb'],
            memory_allocated_mb=memory_stats['allocated_mb'],
            memory_reserved_mb=memory_stats['reserved_mb'],
            cpu_percent=cpu_percent,
            system_memory_mb=system_memory,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            torch_version=torch.__version__,
            device_name=device_info.get('name', str(device))
        )
        
        self.results.append(result)
        tracemalloc.stop()
        
        return result
    
    def run_comprehensive_benchmark(self, models: List[str] = None, 
                                  batch_sizes: List[int] = None,
                                  devices: List[str] = None) -> List[BenchmarkResult]:
        """Run comprehensive benchmarks across multiple configurations."""
        
        models = models or ['cnn', 'transformer', 'diffusion']
        batch_sizes = batch_sizes or [8, 16, 32]
        devices = devices or ['auto']
        
        all_results = []
        
        for device_pref in devices:
            for model_name in models:
                for batch_size in batch_sizes:
                    try:
                        print(f"Benchmarking {model_name} with batch_size={batch_size} on {device_pref}")
                        result = self.benchmark_model(
                            model_name=model_name,
                            batch_size=batch_size,
                            device_preference=device_pref
                        )
                        all_results.append(result)
                        print(f"  Throughput: {result.throughput_samples_per_sec:.2f} samples/sec")
                        print(f"  Peak Memory: {result.peak_memory_mb:.2f} MB")
                        print(f"  Total Time: {result.total_time_ms:.2f} ms")
                        
                    except Exception as e:
                        print(f"Failed to benchmark {model_name} with batch_size={batch_size} on {device_pref}: {e}")
                        continue
        
        return all_results
    
    def save_results(self, output_dir: str = "benchmark_results"):
        """Save benchmark results to JSON and CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_file = output_path / f"benchmark_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        # Save as CSV
        csv_file = output_path / f"benchmark_results_{timestamp}.csv"
        if self.results:
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(asdict(result))
        
        print(f"Results saved to {json_file} and {csv_file}")
        return json_file, csv_file


def create_benchmark_cli():
    """Create command-line interface for benchmarking."""
    parser = argparse.ArgumentParser(description="Feilian GPU Benchmark Suite")
    
    parser.add_argument('--models', nargs='+', choices=['cnn', 'transformer', 'diffusion'],
                       default=['cnn', 'transformer', 'diffusion'],
                       help='Models to benchmark')
    
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[8, 16, 32],
                       help='Batch sizes to test')
    
    parser.add_argument('--devices', nargs='+', choices=['auto', 'cpu', 'mps', 'cuda'],
                       default=['auto'], help='Devices to test')
    
    parser.add_argument('--iterations', type=int, default=50,
                       help='Number of iterations per benchmark')
    
    parser.add_argument('--warmup', type=int, default=10,
                       help='Number of warmup iterations')
    
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Directory to save results')
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_benchmark_cli()
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = BenchmarkRunner()
    
    # Run comprehensive benchmark
    results = runner.run_comprehensive_benchmark(
        models=args.models,
        batch_sizes=args.batch_sizes,
        devices=args.devices
    )
    
    # Save results
    json_file, csv_file = runner.save_results(args.output_dir)
    
    # Print summary
    print(f"\n=== Benchmark Summary ===")
    print(f"Total configurations tested: {len(results)}")
    
    if results:
        best_throughput = max(results, key=lambda r: r.throughput_samples_per_sec)
        print(f"\nBest Performance:")
        print(f"  Model: {best_throughput.model_name}")
        print(f"  Device: {best_throughput.device}")
        print(f"  Batch Size: {best_throughput.batch_size}")
        print(f"  Throughput: {best_throughput.throughput_samples_per_sec:.2f} samples/sec")
        
        lowest_memory = min(results, key=lambda r: r.peak_memory_mb)
        print(f"\nMost Memory Efficient:")
        print(f"  Model: {lowest_memory.model_name}")
        print(f"  Device: {lowest_memory.device}")
        print(f"  Batch Size: {lowest_memory.batch_size}")
        print(f"  Peak Memory: {lowest_memory.peak_memory_mb:.2f} MB")


if __name__ == '__main__':
    main()
