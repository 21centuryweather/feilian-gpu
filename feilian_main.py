"""
Enhanced Feilian GPU Training Script with Cross-Platform GPU Support
Supports Apple Silicon (MPS), NVIDIA CUDA, and CPU training with configurable options.
"""

import csv
import os
import re
import sys
import argparse
import logging
from datetime import datetime
from os import walk

import numpy as np
import torch.nn as nn

from feilian import (
    DataFormatter,
    FeilianNet,
    train_network_model_with_adam,
    predict_with_model,
)
from feilian import get_device_manager, print_available_devices

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PATTERN = re.compile(r"/([^/]+)_deg(\d+)(.?)\\.npy")


def setup_argparse():
    """Setup command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Feilian Wind Flow Prediction Training with GPU Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument("seed", type=int, help="Random seed for reproducible splits")
    parser.add_argument(
        "--data-path",
        default="raw_data/wind3D/idealized/",
        help="Path to training data",
    )
    parser.add_argument(
        "--formatted-shape",
        type=int,
        default=1280,
        help="Formatted shape for data preprocessing",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training",
    )

    # Model arguments
    parser.add_argument(
        "--chan-multi", type=int, default=20, help="Channel multiplier for network"
    )
    parser.add_argument(
        "--max-level", type=int, default=6, help="Maximum level of the U-Net"
    )
    parser.add_argument(
        "--activation",
        choices=["ReLU", "PReLU"],
        default="ReLU",
        help="Activation function to use",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1000, help="Number of training epochs"
    )
    parser.add_argument(
        "--model-dir", default="./models", help="Directory to save models"
    )

    # Device arguments
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device preference for training",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage regardless of available hardware",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        default=True,
        help="Use mixed precision training (CUDA only)",
    )
    parser.add_argument(
        "--no-mixed-precision",
        dest="mixed_precision",
        action="store_false",
        help="Disable mixed precision training",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        default=True,
        help="Save periodic training checkpoints",
    )
    parser.add_argument(
        "--no-checkpoints",
        dest="save_checkpoints",
        action="store_false",
        help="Disable checkpoint saving",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="Epoch interval for saving checkpoints",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir", default=".output", help="Directory for output files"
    )
    parser.add_argument(
        "--save-images", action="store_true", help="Save prediction images"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output including device information",
    )

    return parser


def load_files_from_path(datapath):
    """
    Load .npy and .nc files from a specified path, handling different directory structures.

    Args:
        datapath (str): The base path to search for .npy and .nc files.

    Returns:
        list: A sorted list of file paths for .npy and .nc files found.
    """
    paths = []
    if datapath.startswith("data/all"):
        paths.extend(
            [
                f"data/{x}/{y}/"
                for x in ["uniform", "variable"]
                for y in ["realistic", "idealised"]
            ]
        )
    elif "/all" in datapath:
        paths.extend(
            [
                datapath.replace("/all", "/idealised"),
                datapath.replace("/all", "/realistic"),
            ]
        )
    else:
        paths.append(datapath)

    datafiles = []
    for path in paths:
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            continue
        for _, _, fnames in walk(path):
            # Include both .npy and .nc files
            datafiles.extend(
                [
                    os.path.join(path, fname)
                    for fname in fnames
                    if fname.endswith((".npy", ".nc", ".netcdf"))
                ]
            )
            break

    datafiles.sort()
    logger.info(f"Found {len(datafiles)} data files")
    return datafiles


def parse_wind_angle(filename):
    """Parse the wind angle from the given filename."""
    # First try the original .npy pattern
    rematch = PATTERN.search(filename)
    if rematch:
        return int(rematch.group(2))

    # Try NetCDF filename patterns
    if filename.endswith((".nc", ".netcdf")):
        try:
            from feilian import parse_wind_angle_from_netcdf_filename

            return parse_wind_angle_from_netcdf_filename(filename)
        except ImportError:
            logger.warning(
                "NetCDF support not available, cannot parse wind angle from NetCDF filename"
            )

    return 0


def parse_topo_name(filename):
    """Parse the topology name from the given filename."""
    rematch = PATTERN.search(filename)
    if rematch:
        return rematch.group(1)
    return f"topo{datetime.now().strftime('%H-%M-%S.%f')}"


def write_dicts_to_csv(data, filenames, case_angles, csvname):
    """Write data from dictionaries to a CSV file."""
    topo = [parse_topo_name(fn) for fn in filenames] + ["total"]
    caseangles = case_angles + [0]
    with open(csvname, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["topo", "angle"] + list(data.keys()))
        csv_writer.writerows(zip(topo, caseangles, *data.values()))


def save_training_results(
    data_fmt, files, angles, path, fmt_pred, indices, csv_name, saveimages=False
):
    """Save training results including metrics and optionally images."""
    if not os.path.exists(path):
        os.makedirs(path)

    metrics_csv = f"{path}/{csv_name}"
    metrics = data_fmt.compute_all_metrics(fmt_pred, indices)
    write_dicts_to_csv(
        metrics, [files[i] for i in indices], [angles[i] for i in indices], metrics_csv
    )
    logger.info(f"Metrics saved to {metrics_csv}")

    if not saveimages:
        return

    raw_pred = data_fmt.restore_raw_output_data(fmt_pred, indices, np.nan)
    for pidx, tidx in enumerate(indices):
        images = [np.load(f) for f in files]  # Load images for saving
        pred, truth = np.rot90(raw_pred[pidx], k=-(angles[tidx] // 90)), images[tidx]
        truth[truth < 0] = np.nan
        casename = files[tidx].split("/")[-1].split(".npy")[0]
        np.save(f"{path}/{casename}_truth.npy", truth)
        np.save(f"{path}/{casename}_prediction.npy", pred)
    logger.info(f"Images saved to {path}")


def get_activation_function(activation_name):
    """Get activation function from name."""
    if activation_name == "ReLU":
        return nn.ReLU(inplace=True)
    elif activation_name == "PReLU":
        return nn.PReLU()
    else:
        raise ValueError(f"Unknown activation function: {activation_name}")


def main():
    """Main training function."""
    parser = setup_argparse()
    args = parser.parse_args()

    # Print device information if verbose
    if args.verbose:
        print_available_devices()

    # Initialize device manager
    device_manager = get_device_manager(args.device, args.force_cpu)
    logger.info(f"Using device: {device_manager.device}")

    # Load data files
    logger.info(f"Loading data from: {args.data_path}")
    files = load_files_from_path(args.data_path)

    if not files:
        logger.error("No data files found!")
        sys.exit(1)

    # Load images and angles
    logger.info("Loading images...")
    images = []
    for f in files:
        try:
            # Use universal loader that handles both .npy and .nc files
            if f.endswith((".nc", ".netcdf")):
                try:
                    from feilian import load_wind_data

                    # For NetCDF files, extract z=1 slice and use 'wind_speed' variable
                    data = load_wind_data(
                        f, netcdf_variable="wind_speed", z_level=1, transpose=False
                    )
                    logger.info(f"Loaded NetCDF file {f} with shape {data.shape}")
                    images.append(data)
                except ImportError:
                    logger.error(
                        "NetCDF support not available. Please install netCDF4: pip install netCDF4"
                    )
                    sys.exit(1)
                except Exception:
                    # Try alternative variable names
                    try:
                        from feilian import load_wind_data

                        data = load_wind_data(
                            f, netcdf_variable="Uped", z_level=1, transpose=False
                        )
                        logger.info(
                            f"Loaded NetCDF file {f} using 'Uped' variable with shape {data.shape}"
                        )
                        images.append(data)
                    except Exception as e2:
                        logger.error(f"Failed to load NetCDF file {f}: {e2}")
                        sys.exit(1)
            else:
                # Standard numpy loading for .npy files
                data = np.load(f)
                images.append(data)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            sys.exit(1)

    angles = [parse_wind_angle(fname) for fname in files]
    logger.info(f"Loaded {len(images)} images with angles: {set(angles)}")

    # Initialize data formatter
    logger.info(f"Initializing data formatter with shape: {args.formatted_shape}")
    data_fmt = DataFormatter(
        images, wind_angles=angles, formatted_shape=args.formatted_shape
    )

    # Split data
    x_train, y_train, train_idx, x_test, y_test, test_idx = (
        data_fmt.split_train_test_data(args.train_ratio, args.seed)
    )

    logger.info(f"Data split with seed: {args.seed}")
    logger.info(f"Train indices: {train_idx}")
    logger.info(f"Train data shape: {np.shape(x_train)}")
    logger.info(f"Test indices: {test_idx}")
    logger.info(f"Test data shape: {np.shape(x_test)}")

    # Initialize model
    activation = get_activation_function(args.activation)
    model = FeilianNet(
        chan_multi=args.chan_multi, max_level=args.max_level, activation=activation
    )

    logger.info(f"Model parameters: {model.count_trainable_parameters():,}")
    logger.info(f"Activation: {args.activation}")
    logger.info(f"Batch size: {args.batch_size}")

    # Train model
    logger.info("Starting training...")
    model = train_network_model_with_adam(
        model,
        x_train,
        y_train,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        num_epochs=args.num_epochs,
        model_dir=args.model_dir,
        device_preference=args.device,
        use_mixed_precision=args.mixed_precision,
        save_checkpoints=args.save_checkpoints,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Generate predictions
    logger.info("Generating predictions...")
    y_train_pred = predict_with_model(model, x_train, args.batch_size, device_manager)
    y_test_pred = predict_with_model(model, x_test, args.batch_size, device_manager)

    # Save results
    curr_time = datetime.now().strftime("%Y%m%dT%H:%M:%S.%f")
    case_type = args.data_path.replace("data/", "").rstrip("/")

    # Training results
    train_images_dir = f"{args.output_dir}/images/{case_type}/train"
    train_metrics_csv = (
        f"metrics_train_seed{args.seed}_act{args.activation}_time{curr_time}.csv"
    )
    save_training_results(
        data_fmt,
        files,
        angles,
        train_images_dir,
        y_train_pred,
        train_idx,
        train_metrics_csv,
        args.save_images,
    )

    # Test results
    test_images_dir = f"{args.output_dir}/images/{case_type}/test"
    test_metrics_csv = (
        f"metrics_test_seed{args.seed}_act{args.activation}_time{curr_time}.csv"
    )
    save_training_results(
        data_fmt,
        files,
        angles,
        test_images_dir,
        y_test_pred,
        test_idx,
        test_metrics_csv,
        args.save_images,
    )

    logger.info("Training completed successfully!")

    # Print final device memory usage if CUDA
    if device_manager.device.type == "cuda":
        import torch

        logger.info(
            f"Final GPU memory usage: {torch.cuda.memory_allocated(device_manager.device) / 1e9:.1f} GB"
        )

    # Clear cache
    device_manager.clear_cache()


if __name__ == "__main__":
    main()
