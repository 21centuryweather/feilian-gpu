"""
Tests for distributed training functionality in Feilian-GPU.

This test suite covers:
- Distributed setup and teardown
- Cross-platform backend selection
- DDP model wrapping
- Distributed data loading
- Integration with existing features

Run single-process tests with:
    python -m pytest tests/test_distributed.py

Run distributed tests with:
    torchrun --nproc_per_node=2 --standalone tests/test_distributed.py
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pytest

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feilian.distributed import (
    setup_distributed_training,
    cleanup_distributed,
    create_distributed_sampler,
    wrap_model_for_ddp,
    all_reduce_metrics,
    barrier_and_print,
    distributed_context,
    get_distributed_info,
    DistributedInfo,
    _is_distributed_environment,
    _select_backend,
    _is_apple_silicon,
    _get_device_for_rank,
)
from feilian.device_manager import DeviceManager
from feilian.neural_network import FeilianNet, train_network_model_with_adam
from feilian import DataFormatter


class TestDistributedSetup(unittest.TestCase):
    """Test distributed training setup and configuration."""

    def setUp(self):
        """Clean up any existing distributed state."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def tearDown(self):
        """Clean up distributed state after tests."""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def test_distributed_environment_detection(self):
        """Test detection of distributed training environment variables."""
        # Test without distributed environment
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_is_distributed_environment())

        # Test with partial distributed environment
        with patch.dict(os.environ, {"RANK": "0"}, clear=True):
            self.assertFalse(_is_distributed_environment())

        # Test with complete distributed environment
        with patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "2"}, clear=True):
            self.assertTrue(_is_distributed_environment())

    def test_backend_selection(self):
        """Test automatic backend selection logic."""
        with patch('torch.cuda.is_available', return_value=False):
            with patch('feilian.distributed._is_apple_silicon', return_value=False):
                self.assertEqual(_select_backend(), "gloo")

        with patch('torch.cuda.is_available', return_value=True):
            with patch('feilian.distributed._is_apple_silicon', return_value=False):
                self.assertEqual(_select_backend(), "nccl")

        with patch('torch.cuda.is_available', return_value=True):
            with patch('feilian.distributed._is_apple_silicon', return_value=True):
                self.assertEqual(_select_backend(), "gloo")

    def test_apple_silicon_detection(self):
        """Test Apple Silicon platform detection."""
        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='arm64'):
                self.assertTrue(_is_apple_silicon())

        with patch('platform.system', return_value='Darwin'):
            with patch('platform.machine', return_value='x86_64'):
                self.assertFalse(_is_apple_silicon())

        with patch('platform.system', return_value='Linux'):
            with patch('platform.machine', return_value='arm64'):
                self.assertFalse(_is_apple_silicon())

    def test_device_selection_for_rank(self):
        """Test device assignment based on rank and backend."""
        # Test CUDA device selection
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.set_device') as mock_set_device:
                device = _get_device_for_rank(1, "nccl")
                self.assertEqual(device.type, "cuda")
                self.assertEqual(device.index, 1)
                mock_set_device.assert_called_once()

        # Test MPS device selection
        with patch('torch.backends.mps.is_available', return_value=True):
            with patch('hasattr', return_value=True):
                device = _get_device_for_rank(0, "gloo")
                self.assertEqual(device.type, "mps")

        # Test CPU fallback
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.backends.mps.is_available', return_value=False):
                device = _get_device_for_rank(0, "gloo")
                self.assertEqual(device.type, "cpu")

    def test_single_process_setup(self):
        """Test setup in single-process (non-distributed) mode."""
        with patch.dict(os.environ, {}, clear=True):
            dist_info = setup_distributed_training()
            
            self.assertFalse(dist_info.is_distributed)
            self.assertEqual(dist_info.rank, 0)
            self.assertEqual(dist_info.world_size, 1)
            self.assertTrue(dist_info.is_main_process)

    @patch('torch.distributed.init_process_group')
    @patch('torch.distributed.is_initialized', return_value=True)
    def test_distributed_setup(self, mock_is_init, mock_init):
        """Test distributed setup with mocked torch.distributed."""
        with patch.dict(os.environ, {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}):
            with patch('feilian.distributed._get_device_for_rank', return_value=torch.device('cuda:1')):
                dist_info = setup_distributed_training(backend="nccl")
                
                self.assertTrue(dist_info.is_distributed)
                self.assertEqual(dist_info.rank, 1)
                self.assertEqual(dist_info.local_rank, 1)
                self.assertEqual(dist_info.world_size, 4)
                self.assertEqual(dist_info.backend, "nccl")
                self.assertFalse(dist_info.is_main_process)
                self.assertEqual(dist_info.device, torch.device('cuda:1'))
                
                mock_init.assert_called_once()

    def test_distributed_context_manager(self):
        """Test distributed context manager."""
        with patch.dict(os.environ, {}, clear=True):
            with distributed_context() as dist_info:
                self.assertFalse(dist_info.is_distributed)
                self.assertEqual(dist_info.rank, 0)


class TestDistributedDataLoading(unittest.TestCase):
    """Test distributed data loading and sampling."""

    def setUp(self):
        self.dataset = torch.utils.data.TensorDataset(
            torch.randn(100, 3, 32, 32),
            torch.randn(100, 1, 32, 32)
        )
        self.dist_info = DistributedInfo(
            is_distributed=True,
            rank=0,
            world_size=2,
            backend="gloo",
            is_main_process=True
        )

    def test_distributed_sampler_creation(self):
        """Test creation of distributed sampler."""
        sampler = create_distributed_sampler(self.dataset, self.dist_info, shuffle=True)
        
        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.num_replicas, 2)
        self.assertEqual(sampler.rank, 0)
        self.assertTrue(sampler.shuffle)

    def test_single_process_sampler(self):
        """Test that no sampler is created in single-process mode."""
        single_dist_info = DistributedInfo(is_distributed=False)
        sampler = create_distributed_sampler(self.dataset, single_dist_info)
        
        self.assertIsNone(sampler)


class TestDistributedModelWrapper(unittest.TestCase):
    """Test DDP model wrapping functionality."""

    def setUp(self):
        self.model = nn.Linear(10, 1)
        self.dist_info = DistributedInfo(
            is_distributed=True,
            rank=0,
            local_rank=0,
            world_size=2,
            backend="gloo",
            is_main_process=True,
            device=torch.device('cpu')
        )

    def test_ddp_model_wrapping(self):
        """Test wrapping model with DDP."""
        with patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
            mock_ddp.return_value = self.model
            
            wrapped_model = wrap_model_for_ddp(self.model, self.dist_info)
            
            # Model should be moved to device and wrapped
            mock_ddp.assert_called_once()
            args, kwargs = mock_ddp.call_args
            self.assertEqual(args[0], self.model)

    def test_single_process_no_wrapping(self):
        """Test that model is not wrapped in single-process mode."""
        single_dist_info = DistributedInfo(is_distributed=False)
        wrapped_model = wrap_model_for_ddp(self.model, single_dist_info)
        
        self.assertEqual(wrapped_model, self.model)

    def test_cuda_device_ids(self):
        """Test DDP with CUDA device IDs."""
        cuda_dist_info = DistributedInfo(
            is_distributed=True,
            rank=0,
            local_rank=0,
            world_size=2,
            backend="nccl",
            device=torch.device('cuda:0')
        )
        
        with patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
            wrap_model_for_ddp(self.model, cuda_dist_info)
            
            args, kwargs = mock_ddp.call_args
            self.assertEqual(kwargs['device_ids'], [0])
            self.assertEqual(kwargs['output_device'], 0)


class TestDistributedUtilities(unittest.TestCase):
    """Test distributed utility functions."""

    def setUp(self):
        self.dist_info = DistributedInfo(
            is_distributed=True,
            rank=0,
            world_size=2,
            device=torch.device('cpu')
        )

    @patch('torch.distributed.all_reduce')
    def test_all_reduce_metrics(self, mock_all_reduce):
        """Test metric reduction across processes."""
        metrics = {"loss": 1.0, "accuracy": 0.8}
        
        # Mock the all_reduce operation
        def mock_reduce(tensor, op):
            tensor.fill_(2.0)  # Simulate averaging
        
        mock_all_reduce.side_effect = mock_reduce
        
        reduced_metrics = all_reduce_metrics(metrics, self.dist_info)
        
        self.assertEqual(len(reduced_metrics), 2)
        self.assertEqual(mock_all_reduce.call_count, 2)

    def test_single_process_metrics(self):
        """Test metric reduction in single-process mode."""
        single_dist_info = DistributedInfo(is_distributed=False)
        metrics = {"loss": 1.0, "accuracy": 0.8}
        
        reduced_metrics = all_reduce_metrics(metrics, single_dist_info)
        
        self.assertEqual(reduced_metrics, metrics)

    @patch('torch.distributed.barrier')
    @patch('builtins.print')
    def test_barrier_and_print(self, mock_print, mock_barrier):
        """Test synchronized printing."""
        barrier_and_print("Test message", self.dist_info)
        
        mock_barrier.assert_called_once()
        mock_print.assert_called_once_with("Test message")

    @patch('builtins.print')
    def test_barrier_and_print_single_process(self, mock_print):
        """Test printing in single-process mode."""
        single_dist_info = DistributedInfo(is_distributed=False)
        barrier_and_print("Test message", single_dist_info)
        
        mock_print.assert_called_once_with("Test message")


class TestDeviceManagerIntegration(unittest.TestCase):
    """Test DeviceManager integration with distributed training."""

    def test_device_manager_distributed_init(self):
        """Test DeviceManager initialization with distributed support."""
        with patch('feilian.distributed.setup_distributed_training') as mock_setup:
            mock_setup.return_value = DistributedInfo(
                is_distributed=True,
                rank=0,
                world_size=2,
                device=torch.device('cpu')
            )
            
            dm = DeviceManager(
                device_preference="auto",
                enable_distributed=True,
                distributed_backend="gloo"
            )
            
            self.assertTrue(dm.is_distributed())
            self.assertEqual(dm.get_rank(), 0)
            self.assertEqual(dm.get_world_size(), 2)
            mock_setup.assert_called_once_with("gloo")

    def test_device_manager_single_process(self):
        """Test DeviceManager in single-process mode."""
        dm = DeviceManager(enable_distributed=False)
        
        self.assertFalse(dm.is_distributed())
        self.assertEqual(dm.get_rank(), 0)
        self.assertEqual(dm.get_world_size(), 1)
        self.assertTrue(dm.is_main_process())

    def test_dataloader_creation_distributed(self):
        """Test distributed dataloader creation."""
        dataset = torch.utils.data.TensorDataset(
            torch.randn(20, 3),
            torch.randn(20, 1)
        )
        
        with patch('feilian.distributed.setup_distributed_training') as mock_setup:
            mock_setup.return_value = DistributedInfo(
                is_distributed=True,
                rank=0,
                world_size=2,
                device=torch.device('cpu')
            )
            
            dm = DeviceManager(enable_distributed=True)
            dataloader = dm.create_dataloader(dataset, batch_size=8)
            
            # Should have distributed sampler
            self.assertIsNotNone(dataloader.sampler)
            # Batch size should be divided by world size
            self.assertEqual(dataloader.batch_size, 4)  # 8 / 2


class TestTrainingIntegration(unittest.TestCase):
    """Test integration of distributed training with the full training pipeline."""

    def setUp(self):
        """Set up test data and model."""
        # Create simple synthetic data
        np.random.seed(42)
        images = [np.random.rand(64, 64) for _ in range(4)]
        angles = [0, 90, 180, 270]
        
        self.data_formatter = DataFormatter(images, angles, formatted_shape=64)
        self.x_train, self.y_train, _, _, _, _ = self.data_formatter.split_train_test_data(0.8, 42)
        
        # Create small model for testing
        self.model = FeilianNet(
            chan_multi=4,
            max_level=2,
            activation=nn.ReLU()
        )

    def test_training_with_distributed_disabled(self):
        """Test normal training without distributed training."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trained_model = train_network_model_with_adam(
                self.model,
                self.x_train,
                self.y_train,
                batch_size=2,
                num_epochs=2,
                model_dir=temp_dir,
                enable_distributed=False
            )
            
            self.assertIsNotNone(trained_model)
            # Check that model file was saved
            model_files = [f for f in os.listdir(temp_dir) if f.endswith('.pth')]
            self.assertGreater(len(model_files), 0)

    @patch('feilian.distributed.setup_distributed_training')
    def test_training_with_distributed_enabled(self, mock_setup):
        """Test training with distributed training enabled."""
        mock_setup.return_value = DistributedInfo(
            is_distributed=True,
            rank=0,
            world_size=2,
            device=torch.device('cpu'),
            is_main_process=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            trained_model = train_network_model_with_adam(
                self.model,
                self.x_train,
                self.y_train,
                batch_size=2,
                num_epochs=2,
                model_dir=temp_dir,
                enable_distributed=True,
                distributed_backend="gloo"
            )
            
            self.assertIsNotNone(trained_model)
            mock_setup.assert_called_once_with("gloo")

    @patch('feilian.distributed.setup_distributed_training')
    def test_worker_process_no_saving(self, mock_setup):
        """Test that worker processes don't save models."""
        mock_setup.return_value = DistributedInfo(
            is_distributed=True,
            rank=1,  # Worker process
            world_size=2,
            device=torch.device('cpu'),
            is_main_process=False
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            train_network_model_with_adam(
                self.model,
                self.x_train,
                self.y_train,
                batch_size=2,
                num_epochs=2,
                model_dir=temp_dir,
                enable_distributed=True
            )
            
            # Worker process should not save model files
            model_files = [f for f in os.listdir(temp_dir) if f.endswith('.pth')]
            self.assertEqual(len(model_files), 0)


class TestDistributedEndToEnd(unittest.TestCase):
    """End-to-end tests for distributed training (require actual distributed setup)."""

    @pytest.mark.skipif(not _is_distributed_environment(), 
                       reason="Requires distributed environment (torchrun)")
    def test_actual_distributed_setup(self):
        """Test actual distributed setup when run with torchrun."""
        dist_info = setup_distributed_training()
        
        self.assertTrue(dist_info.is_distributed)
        self.assertGreaterEqual(dist_info.rank, 0)
        self.assertGreater(dist_info.world_size, 1)
        
        # Test cleanup
        cleanup_distributed()


def run_distributed_tests():
    """Run tests that require actual distributed setup."""
    if _is_distributed_environment():
        # We're running in a distributed environment
        dist_info = setup_distributed_training()
        
        if dist_info.is_main_process:
            print(f"Running distributed tests with {dist_info.world_size} processes")
        
        # Run simple distributed test
        test_model = nn.Linear(10, 1)
        wrapped_model = wrap_model_for_ddp(test_model, dist_info)
        
        if dist_info.is_main_process:
            print(f"✓ Model successfully wrapped with DDP on rank {dist_info.rank}")
        
        # Test metric reduction
        metrics = {"test_loss": float(dist_info.rank + 1)}
        reduced_metrics = all_reduce_metrics(metrics, dist_info)
        
        if dist_info.is_main_process:
            expected_avg = sum(range(dist_info.world_size)) / dist_info.world_size + 1
            print(f"✓ Metrics reduced: {reduced_metrics['test_loss']:.2f} (expected: {expected_avg:.2f})")
        
        # Synchronized barrier and cleanup
        barrier_and_print("All processes completed successfully", dist_info)
        cleanup_distributed()
        
        if dist_info.is_main_process:
            print("✓ All distributed tests passed")
    else:
        print("Running in single-process mode - distributed tests skipped")
        print("To run distributed tests, use: torchrun --nproc_per_node=2 --standalone tests/test_distributed.py")


if __name__ == "__main__":
    if _is_distributed_environment():
        # Running with torchrun - run distributed-specific tests
        run_distributed_tests()
    else:
        # Running normally - run unit tests
        unittest.main()