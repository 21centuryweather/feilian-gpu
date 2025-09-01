"""
NetCDF Data Loader for Feilian Wind Flow Prediction
===================================================

This module provides comprehensive utilities for loading and processing 3D NetCDF
wind speed data, with automatic z-level extraction, NaN handling, and robust error
management. It seamlessly integrates with the existing 2D NumPy workflow while
adding support for complex 3D atmospheric data.

Key Features:
- 3D to 2D wind data extraction with configurable z-levels
- Automatic NaN detection and replacement for training stability
- Robust error handling for corrupted or malformed NetCDF files
- Universal data loader supporting both .nc and .npy formats
- Batch conversion utilities for large datasets
- Wind angle parsing from filename patterns
- Comprehensive file type detection and validation

Supported Formats:
- NetCDF (.nc, .nc4, .netcdf): 3D atmospheric data with multiple variables
- NumPy (.npy, .npz): Pre-processed 2D wind speed arrays

Data Processing Pipeline:
1. File format detection and validation
2. Variable identification (wind_speed, Uped, velocity, etc.)
3. Z-level extraction (typically z=0 for surface, z=1 for ground level)
4. NaN value detection and replacement with zeros
5. Data type normalization to float32
6. Optional transposition for coordinate system alignment

Typical Usage:
    >>> # Load single NetCDF file
    >>> wind_data = load_netcdf_wind_speed('data.nc', z_level=1)
    >>>
    >>> # Universal loader for any format
    >>> data = load_wind_data('data.nc')  # or 'data.npy'
    >>>
    >>> # Batch convert directory
    >>> converted = batch_convert_netcdf_to_numpy('raw_data/', 'processed/')

Author: Feilian Development Team
Version: 1.0.0
"""

import os
import re
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

# Configure module logger
logger = logging.getLogger(__name__)

# ============================================================================
# NetCDF Dependency Management
# ============================================================================

# Check for xarray availability (preferred)
try:
    import xarray as xr

    XARRAY_AVAILABLE = True
    logger.info("xarray support available (preferred for NetCDF handling)")
except ImportError:
    XARRAY_AVAILABLE = False
    xr = None
    logger.info("xarray not available, falling back to netCDF4")

# Check for netCDF4 availability (fallback)
try:
    import netCDF4 as nc

    NETCDF4_AVAILABLE = True
    if not XARRAY_AVAILABLE:
        logger.info("netCDF4 support available (fallback mode)")
except ImportError:
    NETCDF4_AVAILABLE = False
    nc = None
    if not XARRAY_AVAILABLE:
        logger.warning(
            "Neither xarray nor netCDF4 available. Install with: pip install xarray netcdf4"
        )

# Overall NetCDF support availability
NETCDF_AVAILABLE = XARRAY_AVAILABLE or NETCDF4_AVAILABLE


def is_netcdf_available() -> bool:
    """Check if NetCDF4 library is available."""
    return NETCDF_AVAILABLE


def load_netcdf_wind_speed_xarray(
    filepath: str,
    variable_name: str = "wind_speed",
    z_level: int = 1,
    transpose: bool = False,
) -> np.ndarray:
    """
    Load wind speed data from 3D NetCDF file using xarray (preferred method).

    This function provides a more robust and feature-rich approach to loading
    NetCDF files compared to the direct netCDF4 approach. It handles:
    - Automatic dimension detection and labeling
    - Lazy loading for memory efficiency
    - Better metadata handling
    - More intuitive data selection

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    variable_name : str, optional
        Name of the wind speed variable in the NetCDF file (default: 'wind_speed')
    z_level : int, optional
        Z-level to extract (default: 1 for ground level)
    transpose : bool, optional
        Whether to transpose the resulting 2D array (default: False)

    Returns
    -------
    numpy.ndarray
        2D wind speed data at the specified z-level as float32

    Raises
    ------
    ImportError
        If xarray is not installed
    FileNotFoundError
        If the NetCDF file doesn't exist
    KeyError
        If the specified variable is not found in the file
    ValueError
        If the data doesn't have the expected dimensions

    Notes
    -----
    This function uses xarray for enhanced NetCDF handling, providing:
    - Automatic dimension coordinate handling
    - Better memory efficiency through lazy loading
    - More robust error handling and metadata access
    - Seamless integration with the scientific Python ecosystem

    Examples
    --------
    >>> # Load ground-level wind data
    >>> wind_data = load_netcdf_wind_speed_xarray('wind_field.nc', z_level=1)
    >>> print(wind_data.shape)  # (130, 130)
    >>>
    >>> # Load specific variable with transposition
    >>> wind_data = load_netcdf_wind_speed_xarray(
    ...     'simulation.nc',
    ...     variable_name='Uped',
    ...     z_level=0,
    ...     transpose=True
    ... )
    """
    if not XARRAY_AVAILABLE:
        raise ImportError(
            "xarray library is required. Install with: pip install xarray"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    logger.debug(f"Loading NetCDF file with xarray: {filepath}")

    try:
        # Open dataset with xarray (lazy loading by default)
        with xr.open_dataset(filepath, decode_times=False) as ds:
            # Log dataset information
            logger.debug(f"Dataset variables: {list(ds.data_vars.keys())}")
            logger.debug(f"Dataset dimensions: {list(ds.sizes.keys())}")
            logger.debug(f"Dataset coordinates: {list(ds.coords.keys())}")

            # Try to find the wind speed variable
            if variable_name not in ds.data_vars:
                # Common alternative variable names to try
                alt_names = [
                    "wind_speed",
                    "Uped",
                    "u",
                    "velocity",
                    "speed",
                    "ws",
                    "U",
                    "V",
                ]
                found_var = None

                for alt_name in alt_names:
                    if alt_name in ds.data_vars:
                        found_var = alt_name
                        break

                if found_var is None:
                    available_vars = list(ds.data_vars.keys())
                    raise KeyError(
                        f"Variable '{variable_name}' not found. Available variables: {available_vars}"
                    )

                variable_name = found_var
                logger.info(f"Using variable '{variable_name}' for wind speed data")

            # Get the data variable
            data_var = ds[variable_name]
            logger.debug(f"Variable shape: {data_var.shape}")
            logger.debug(f"Variable dimensions: {data_var.dims}")

            # Handle different data structures based on dimensions
            if data_var.ndim == 2:
                # Already 2D data
                result = data_var.values
                logger.info("Data is already 2D, using as-is")

            elif data_var.ndim == 3:
                # 3D data - need to extract z-level slice
                # Special handling for files with duplicate dimension names

                logger.debug(f"3D data detected with dimensions: {data_var.dims}")

                # Check if we have duplicate dimension names (common issue with these NetCDF files)
                dim_names = list(data_var.dims)
                has_duplicate_dims = len(dim_names) != len(set(dim_names))

                if has_duplicate_dims:
                    logger.warning(f"Duplicate dimension names detected: {dim_names}")
                    # For files with duplicate dimension names like ('z', 'z', 'z'),
                    # we need to handle this carefully by using positional indexing
                    # Assume the data is (z, y, x) or (x, y, z) based on common patterns

                    # Try to extract a 2D slice from the 3D data
                    # Use the z_level as index for the first dimension (most common pattern)
                    if z_level < data_var.shape[0]:
                        try:
                            # Use positional indexing instead of dimension names
                            result = data_var.values[z_level, :, :]
                            logger.info(
                                f"Extracted z={z_level} slice using positional indexing (first dimension)"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract slice with positional indexing: {e}"
                            )
                            # Fallback: try other dimension patterns
                            if z_level < data_var.shape[1]:
                                result = data_var.values[:, z_level, :]
                                logger.info(
                                    f"Extracted z={z_level} slice using second dimension"
                                )
                            elif z_level < data_var.shape[2]:
                                result = data_var.values[:, :, z_level]
                                logger.info(
                                    f"Extracted z={z_level} slice using third dimension"
                                )
                            else:
                                # Use first slice as fallback
                                result = data_var.values[0, :, :]
                                logger.warning("Using first slice as fallback")
                    else:
                        max_z = data_var.shape[0] - 1
                        logger.warning(
                            f"z_level {z_level} exceeds maximum {max_z}, using {max_z}"
                        )
                        result = data_var.values[max_z, :, :]

                else:
                    # Normal case with distinct dimension names
                    z_dim_names = [
                        "z",
                        "height",
                        "level",
                        "vertical",
                        "lev",
                        "alt",
                        "altitude",
                    ]
                    z_dim = None

                    # Find the z dimension
                    for dim_name in data_var.dims:
                        if dim_name.lower() in z_dim_names:
                            z_dim = dim_name
                            break

                    if z_dim is not None:
                        # Use xarray's selection capabilities
                        if z_level < data_var.sizes[z_dim]:
                            result = data_var.isel({z_dim: z_level}).values
                            logger.info(
                                f"Extracted z={z_level} slice using dimension '{z_dim}'"
                            )
                        else:
                            max_z = data_var.sizes[z_dim] - 1
                            logger.warning(
                                f"z_level {z_level} exceeds maximum {max_z}, using {max_z}"
                            )
                            result = data_var.isel({z_dim: max_z}).values
                    else:
                        # Assume the first dimension is z if no clear dimension names
                        logger.warning(
                            "No clear z dimension found, assuming first dimension is z"
                        )
                        if z_level < data_var.shape[0]:
                            result = data_var.isel({data_var.dims[0]: z_level}).values
                            logger.info(
                                f"Extracted z={z_level} slice from first dimension"
                            )
                        else:
                            max_z = data_var.shape[0] - 1
                            logger.warning(
                                f"z_level {z_level} exceeds maximum {max_z}, using {max_z}"
                            )
                            result = data_var.isel({data_var.dims[0]: max_z}).values

            elif data_var.ndim == 4:
                # 4D data (time, z, y, x) - take first time step and specified z-level
                logger.info("4D data detected, extracting first time step and z-level")

                # Find time and z dimensions
                time_dim = None
                z_dim = None

                time_dim_names = ["time", "t", "step"]
                z_dim_names = [
                    "z",
                    "height",
                    "level",
                    "vertical",
                    "lev",
                    "alt",
                    "altitude",
                ]

                for dim_name in data_var.dims:
                    if dim_name.lower() in time_dim_names and time_dim is None:
                        time_dim = dim_name
                    elif dim_name.lower() in z_dim_names and z_dim is None:
                        z_dim = dim_name

                # Select first time step and specified z-level
                selection = {}
                if time_dim is not None:
                    selection[time_dim] = 0
                if z_dim is not None and z_level < data_var.sizes[z_dim]:
                    selection[z_dim] = z_level
                elif z_dim is not None:
                    max_z = data_var.sizes[z_dim] - 1
                    selection[z_dim] = max_z
                    logger.warning(
                        f"z_level {z_level} exceeds maximum {max_z}, using {max_z}"
                    )

                if not selection:
                    # Fallback to first two dimensions
                    logger.warning(
                        "Could not identify time/z dimensions, using first slice"
                    )
                    result = data_var.isel(
                        {data_var.dims[0]: 0, data_var.dims[1]: z_level}
                    ).values
                else:
                    result = data_var.isel(selection).values

            else:
                raise ValueError(
                    f"Unsupported data dimensions: {data_var.ndim}D. Expected 2D, 3D, or 4D data."
                )

            # Convert to numpy array and ensure it's float32
            result = np.array(result, dtype=np.float32)

            # Validate that we have a 2D result
            if result.ndim != 2:
                raise ValueError(
                    f"Expected 2D result after extraction, got {result.ndim}D with shape {result.shape}"
                )

            # Check for empty or scalar results
            if result.size == 0:
                raise ValueError(
                    "Extracted data is empty. File might be corrupted or have wrong structure."
                )

            if result.shape == ():
                raise ValueError(
                    "Extracted data is scalar. Expected 2D array but got single value."
                )

            # Handle NaN values - replace with 0 to prevent training issues
            nan_count = np.sum(np.isnan(result))
            if nan_count > 0:
                logger.info(
                    f"Replacing {nan_count} NaN values with 0.0 to prevent training issues"
                )
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply transpose if requested
            if transpose:
                result = result.T
                logger.debug("Applied transpose to data")

            logger.debug(f"Final data shape: {result.shape}")
            logger.info(f"Successfully loaded 2D wind data with shape {result.shape}")
            return result

    except Exception as e:
        logger.error(f"Error loading NetCDF file {filepath} with xarray: {e}")
        raise


def load_netcdf_wind_speed_netcdf4(
    filepath: str,
    variable_name: str = "wind_speed",
    z_level: int = 1,
    transpose: bool = False,
) -> np.ndarray:
    """
    Load wind speed data from 3D NetCDF file using netCDF4 (fallback method).

    This is the original implementation using netCDF4 directly. It's maintained
    for backward compatibility and as a fallback when xarray is not available.

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    variable_name : str, optional
        Name of the wind speed variable in the NetCDF file (default: 'wind_speed')
    z_level : int, optional
        Z-level to extract (default: 1 for ground level)
    transpose : bool, optional
        Whether to transpose the resulting 2D array (default: False)

    Returns
    -------
    numpy.ndarray
        2D wind speed data at the specified z-level as float32

    Raises
    ------
    ImportError
        If netCDF4 is not installed
    FileNotFoundError
        If the NetCDF file doesn't exist
    KeyError
        If the specified variable is not found in the file
    ValueError
        If the data doesn't have the expected dimensions
    """
    if not NETCDF4_AVAILABLE:
        raise ImportError(
            "netCDF4 library is required. Install with: pip install netcdf4"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    logger.debug(f"Loading NetCDF file with netCDF4: {filepath}")

    try:
        with nc.Dataset(filepath, "r") as dataset:
            # Try to find the wind speed variable
            if variable_name not in dataset.variables:
                # Common alternative variable names to try
                alt_names = ["wind_speed", "Uped", "u", "velocity", "speed", "ws"]
                found_var = None

                for alt_name in alt_names:
                    if alt_name in dataset.variables:
                        found_var = alt_name
                        break

                if found_var is None:
                    available_vars = list(dataset.variables.keys())
                    raise KeyError(
                        f"Variable '{variable_name}' not found. Available variables: {available_vars}"
                    )

                variable_name = found_var
                logger.info(f"Using variable '{variable_name}' for wind speed data")

            # Load the variable data
            wind_data = dataset.variables[variable_name]

            # Try to load the data with error handling for corrupted files
            try:
                data_array = wind_data[:]
            except Exception as e:
                # If full load fails, the file might be corrupted
                # Try to access just a slice to check if it's accessible
                try:
                    test_slice = (
                        wind_data[0, :, :]
                        if len(wind_data.shape) >= 3
                        else wind_data[:]
                    )
                    logger.warning(
                        f"File {filepath} has access issues, attempting slice-by-slice loading"
                    )

                    # For 3D data, try to construct the full array slice by slice
                    if len(wind_data.shape) == 3:
                        # We only need the first slice anyway for z=0
                        data_array = test_slice
                        logger.info(
                            "Using z=0 slice directly due to file access issues"
                        )
                    else:
                        data_array = test_slice
                except Exception as e2:
                    logger.error(f"Cannot access data in {filepath}: {e2}")
                    raise ValueError(
                        f"Corrupted NetCDF file: cannot access wind speed data. Original error: {e}"
                    )

            logger.debug(f"Original data shape: {data_array.shape}")
            logger.debug(f"Variable dimensions: {wind_data.dimensions}")

            # Handle different data structures
            if data_array.ndim == 2:
                # Already 2D data
                result = data_array
                logger.info("Data is already 2D, using as-is")

            elif data_array.ndim == 3:
                # 3D data - need to extract z-level slice
                # Try to determine which axis is z
                dims = wind_data.dimensions

                # Common patterns for dimension ordering
                if any(dim in ["z", "height", "level", "vertical"] for dim in dims):
                    # Find the z dimension index
                    z_dim_idx = None
                    for i, dim in enumerate(dims):
                        if dim in ["z", "height", "level", "vertical"]:
                            z_dim_idx = i
                            break

                    if z_dim_idx == 0:
                        result = data_array[z_level, :, :]
                    elif z_dim_idx == 1:
                        result = data_array[:, z_level, :]
                    elif z_dim_idx == 2:
                        result = data_array[:, :, z_level]
                    else:
                        raise ValueError(f"Unexpected z dimension index: {z_dim_idx}")

                else:
                    # Assume the first dimension is z if no clear dimension names
                    logger.warning(
                        "No clear z dimension found, assuming first dimension is z"
                    )
                    result = data_array[z_level, :, :]

                logger.info(f"Extracted z={z_level} slice from 3D data")

            else:
                raise ValueError(
                    f"Unsupported data dimensions: {data_array.ndim}D. Expected 2D or 3D data."
                )

            # Convert to numpy array and ensure it's float32
            result = np.array(result, dtype=np.float32)

            # Handle NaN values - replace with 0 to prevent training issues
            nan_count = np.sum(np.isnan(result))
            if nan_count > 0:
                logger.info(
                    f"Replacing {nan_count} NaN values with 0.0 to prevent training issues"
                )
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply transpose if requested
            if transpose:
                result = result.T
                logger.debug("Applied transpose to data")

            logger.debug(f"Final data shape: {result.shape}")
            return result

    except Exception as e:
        logger.error(f"Error loading NetCDF file {filepath} with netCDF4: {e}")
        raise


def load_netcdf_wind_speed(
    filepath: str,
    variable_name: str = "wind_speed",
    z_level: int = 1,
    transpose: bool = False,
    prefer_xarray: bool = True,
) -> np.ndarray:
    """
    Load wind speed data from 3D NetCDF file and extract 2D slice at z=z_level.

    This function intelligently chooses between xarray (preferred) and netCDF4 (fallback)
    based on availability and performance considerations. xarray provides better memory
    efficiency, more robust dimension handling, and enhanced metadata support.

    IMPORTANT: For corrected NetCDF files with proper (x, y, z) dimensions:
    - z_level=1 is the recommended ground level with meaningful wind speeds
    - z_level=0 typically contains all zeros (surface level)

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    variable_name : str, optional
        Name of the wind speed variable in the NetCDF file (default: 'wind_speed')
    z_level : int, optional
        Z-level to extract (default: 1 for ground level)
    transpose : bool, optional
        Whether to transpose the resulting 2D array (default: False)
    prefer_xarray : bool, optional
        Whether to prefer xarray over netCDF4 when both are available (default: True)

    Returns
    -------
    numpy.ndarray
        2D wind speed data at the specified z-level as float32

    Raises
    ------
    ImportError
        If neither xarray nor netCDF4 is installed
    FileNotFoundError
        If the NetCDF file doesn't exist
    KeyError
        If the specified variable is not found in the file
    ValueError
        If the data doesn't have the expected dimensions

    Notes
    -----
    Library Selection Priority:
    1. xarray (preferred): Better memory efficiency, dimension handling, metadata support
    2. netCDF4 (fallback): Direct access, faster for simple operations

    The function automatically handles:
    - 2D, 3D, and 4D NetCDF datasets
    - Multiple variable name conventions
    - Dimension detection and coordinate selection
    - NaN value replacement for training stability
    - Memory-efficient lazy loading (with xarray)

    Examples
    --------
    >>> # Load with automatic library selection (xarray preferred)
    >>> wind_data = load_netcdf_wind_speed('wind_field.nc', z_level=1)
    >>>
    >>> # Force netCDF4 usage
    >>> wind_data = load_netcdf_wind_speed('wind_field.nc', prefer_xarray=False)
    >>>
    >>> # Load specific variable with transposition
    >>> wind_data = load_netcdf_wind_speed(
    ...     'simulation.nc', variable_name='Uped', transpose=True
    ... )
    """
    if not NETCDF_AVAILABLE:
        raise ImportError(
            "Neither xarray nor netCDF4 is available. "
            "Install with: pip install xarray netcdf4"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    # Choose the appropriate loader based on availability and preference
    if prefer_xarray and XARRAY_AVAILABLE:
        logger.debug(f"Using xarray for NetCDF loading (preferred): {filepath}")
        try:
            return load_netcdf_wind_speed_xarray(
                filepath, variable_name, z_level, transpose
            )
        except Exception as e:
            if NETCDF4_AVAILABLE:
                logger.warning(f"xarray failed ({e}), falling back to netCDF4")
                return load_netcdf_wind_speed_netcdf4(
                    filepath, variable_name, z_level, transpose
                )
            else:
                raise

    elif NETCDF4_AVAILABLE:
        logger.debug(f"Using netCDF4 for NetCDF loading: {filepath}")
        try:
            return load_netcdf_wind_speed_netcdf4(
                filepath, variable_name, z_level, transpose
            )
        except Exception as e:
            if XARRAY_AVAILABLE:
                logger.warning(f"netCDF4 failed ({e}), falling back to xarray")
                return load_netcdf_wind_speed_xarray(
                    filepath, variable_name, z_level, transpose
                )
            else:
                raise

    elif XARRAY_AVAILABLE:
        logger.debug(f"Using xarray for NetCDF loading (only option): {filepath}")
        return load_netcdf_wind_speed_xarray(
            filepath, variable_name, z_level, transpose
        )

    else:
        raise ImportError(
            "Neither xarray nor netCDF4 is available. "
            "Install with: pip install xarray netcdf4"
        )


def detect_file_type(filepath: str) -> str:
    """
    Detect whether a file is a NetCDF file or numpy file based on extension.

    Args:
        filepath (str): Path to the file

    Returns:
        str: 'netcdf' or 'numpy' or 'unknown'
    """
    _, ext = os.path.splitext(filepath.lower())

    if ext in [".nc", ".netcdf", ".nc4"]:
        return "netcdf"
    elif ext in [".npy", ".npz"]:
        return "numpy"
    else:
        return "unknown"


def load_wind_data(
    filepath: str,
    netcdf_variable: str = "wind_speed",
    z_level: int = 1,
    transpose: bool = False,
) -> np.ndarray:
    """
    Universal data loader that can handle both NetCDF and numpy files.

    For NetCDF files, extracts the z=z_level slice from 3D data.
    For numpy files, loads directly.

    Args:
        filepath (str): Path to the data file
        netcdf_variable (str): Variable name for NetCDF files
        z_level (int): Z-level to extract for NetCDF files
        transpose (bool): Whether to transpose the data

    Returns:
        np.ndarray: 2D wind speed data
    """
    file_type = detect_file_type(filepath)

    if file_type == "netcdf":
        return load_netcdf_wind_speed(filepath, netcdf_variable, z_level, transpose)

    elif file_type == "numpy":
        data = np.load(filepath)
        if transpose:
            data = data.T
        return data.astype(np.float32)

    else:
        raise ValueError(f"Unsupported file type for {filepath}. Supported: .nc, .npy")


def parse_wind_angle_from_netcdf_filename(filename: str) -> int:
    """
    Parse wind angle from NetCDF filename.
    Supports patterns like: case_name_deg180.nc, case_name_180deg.nc, case_name_180.nc

    Args:
        filename (str): NetCDF filename

    Returns:
        int: Wind angle in degrees, or 0 if not found
    """
    # Try different patterns
    patterns = [
        r"_deg(\d+)\.nc",  # case_deg180.nc
        r"_(\d+)deg\.nc",  # case_180deg.nc
        r"_(\d+)\.nc",  # case_180.nc
        r"deg(\d+)",  # any_deg180_anything
        r"(\d+)deg",  # any_180deg_anything
    ]

    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            angle = int(match.group(1))
            # Ensure angle is in valid range
            return angle % 360

    logger.warning(f"Could not parse wind angle from filename: {filename}")
    return 0


def find_wind_data_files(
    data_path: str, include_netcdf: bool = True, include_numpy: bool = True
) -> List[str]:
    """
    Find all wind data files (NetCDF and/or numpy) in a directory.

    Args:
        data_path (str): Directory path to search
        include_netcdf (bool): Include .nc files
        include_numpy (bool): Include .npy files

    Returns:
        List[str]: Sorted list of file paths
    """
    if not os.path.exists(data_path):
        logger.warning(f"Path does not exist: {data_path}")
        return []

    found_files = []

    for root, dirs, files in os.walk(data_path):
        for file in files:
            filepath = os.path.join(root, file)
            file_type = detect_file_type(file)

            if file_type == "netcdf" and include_netcdf:
                found_files.append(filepath)
            elif file_type == "numpy" and include_numpy:
                found_files.append(filepath)

    found_files.sort()
    logger.info(f"Found {len(found_files)} data files in {data_path}")

    return found_files


def convert_netcdf_to_numpy(
    netcdf_path: str,
    output_dir: str,
    variable_name: str = "wind_speed",
    z_level: int = 0,
    transpose: bool = False,
) -> str:
    """
    Convert a NetCDF file to numpy format for compatibility with existing workflow.

    Args:
        netcdf_path (str): Path to NetCDF file
        output_dir (str): Directory to save numpy file
        variable_name (str): Variable name in NetCDF file
        z_level (int): Z-level to extract
        transpose (bool): Whether to transpose the data

    Returns:
        str: Path to the created numpy file
    """
    if not NETCDF_AVAILABLE:
        raise ImportError(
            "NetCDF4 library is required. Install with: pip install netCDF4"
        )

    # Load the data
    data = load_netcdf_wind_speed(netcdf_path, variable_name, z_level, transpose)

    # Create output filename
    base_name = os.path.basename(netcdf_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"{name_without_ext}_z{z_level}.npy"
    output_path = os.path.join(output_dir, output_filename)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save as numpy file
    np.save(output_path, data)
    logger.info(f"Converted {netcdf_path} to {output_path}")

    return output_path


def batch_convert_netcdf_to_numpy(
    netcdf_dir: str,
    output_dir: str,
    variable_name: str = "wind_speed",
    z_level: int = 0,
    transpose: bool = False,
) -> List[str]:
    """
    Convert all NetCDF files in a directory to numpy format.

    Args:
        netcdf_dir (str): Directory containing NetCDF files
        output_dir (str): Directory to save converted numpy files
        variable_name (str): Variable name in NetCDF files
        z_level (int): Z-level to extract
        transpose (bool): Whether to transpose the data

    Returns:
        List[str]: List of paths to created numpy files
    """
    netcdf_files = find_wind_data_files(
        netcdf_dir, include_netcdf=True, include_numpy=False
    )
    converted_files = []

    for netcdf_file in netcdf_files:
        try:
            output_file = convert_netcdf_to_numpy(
                netcdf_file, output_dir, variable_name, z_level, transpose
            )
            converted_files.append(output_file)
        except Exception as e:
            logger.error(f"Failed to convert {netcdf_file}: {e}")

    logger.info(f"Successfully converted {len(converted_files)} NetCDF files")
    return converted_files


# ============================================================================
# Advanced xarray-specific functions
# ============================================================================


def get_netcdf_info(filepath: str) -> Dict[str, Any]:
    """
    Get comprehensive information about a NetCDF file structure.

    This function provides detailed metadata about NetCDF files, which is
    particularly useful for understanding the structure of new datasets.

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file

    Returns
    -------
    Dict[str, Any]
        Dictionary containing file metadata including variables, dimensions,
        coordinates, and attributes

    Raises
    ------
    ImportError
        If neither xarray nor netCDF4 is available
    FileNotFoundError
        If the NetCDF file doesn't exist

    Examples
    --------
    >>> info = get_netcdf_info('wind_data.nc')
    >>> print(info['variables'])  # Available data variables
    >>> print(info['dimensions'])  # Dimension information
    """
    if not NETCDF_AVAILABLE:
        raise ImportError(
            "NetCDF support is required. Install with: pip install xarray netcdf4"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    if XARRAY_AVAILABLE:
        # Use xarray for comprehensive metadata extraction
        with xr.open_dataset(filepath, decode_times=False) as ds:
            info = {
                "variables": {
                    name: {
                        "shape": var.shape,
                        "dimensions": var.dims,
                        "dtype": str(var.dtype),
                        "attributes": dict(var.attrs),
                    }
                    for name, var in ds.data_vars.items()
                },
                "coordinates": {
                    name: {
                        "shape": coord.shape,
                        "dtype": str(coord.dtype),
                        "values_range": (
                            float(coord.min().values) if coord.size > 0 else None,
                            float(coord.max().values) if coord.size > 0 else None,
                        ),
                        "attributes": dict(coord.attrs),
                    }
                    for name, coord in ds.coords.items()
                },
                "dimensions": dict(ds.dims),
                "global_attributes": dict(ds.attrs),
                "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
            }
    else:
        # Fallback to netCDF4 for basic metadata
        with nc.Dataset(filepath, "r") as dataset:
            info = {
                "variables": {
                    name: {
                        "shape": var.shape,
                        "dimensions": var.dimensions,
                        "dtype": str(var.dtype),
                        "attributes": {
                            attr: getattr(var, attr) for attr in var.ncattrs()
                        },
                    }
                    for name, var in dataset.variables.items()
                    if name not in dataset.dimensions
                },
                "coordinates": {
                    name: {
                        "shape": var.shape,
                        "dtype": str(var.dtype),
                        "attributes": {
                            attr: getattr(var, attr) for attr in var.ncattrs()
                        },
                    }
                    for name, var in dataset.variables.items()
                    if name in dataset.dimensions
                },
                "dimensions": dict(dataset.dimensions),
                "global_attributes": {
                    attr: getattr(dataset, attr) for attr in dataset.ncattrs()
                },
                "file_size_mb": os.path.getsize(filepath) / (1024 * 1024),
            }

    return info


def load_multiple_variables(
    filepath: str, variable_names: List[str], z_level: int = 1, transpose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Load multiple variables from a NetCDF file simultaneously.

    This function is useful when you need to load multiple related variables
    (e.g., u and v wind components) from the same NetCDF file efficiently.

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    variable_names : List[str]
        List of variable names to load
    z_level : int, optional
        Z-level to extract (default: 1 for ground level)
    transpose : bool, optional
        Whether to transpose the resulting 2D arrays (default: False)

    Returns
    -------
    Dict[str, numpy.ndarray]
        Dictionary mapping variable names to their 2D data arrays

    Raises
    ------
    ImportError
        If neither xarray nor netCDF4 is available
    FileNotFoundError
        If the NetCDF file doesn't exist
    KeyError
        If any of the specified variables is not found in the file

    Examples
    --------
    >>> # Load u and v wind components
    >>> data = load_multiple_variables('wind_field.nc', ['u', 'v'], z_level=1)
    >>> u_wind = data['u']
    >>> v_wind = data['v']
    >>>
    >>> # Compute wind speed and direction
    >>> wind_speed = np.sqrt(u_wind**2 + v_wind**2)
    >>> wind_direction = np.arctan2(v_wind, u_wind) * 180 / np.pi
    """
    if not NETCDF_AVAILABLE:
        raise ImportError(
            "NetCDF support is required. Install with: pip install xarray netcdf4"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    results = {}

    if XARRAY_AVAILABLE:
        # Use xarray for efficient multi-variable loading
        with xr.open_dataset(filepath, decode_times=False) as ds:
            for var_name in variable_names:
                if var_name not in ds.data_vars:
                    available_vars = list(ds.data_vars.keys())
                    raise KeyError(
                        f"Variable '{var_name}' not found. Available variables: {available_vars}"
                    )

                # Load the variable using the same logic as single variable loading
                try:
                    data = load_netcdf_wind_speed_xarray(
                        filepath, var_name, z_level, transpose
                    )
                    results[var_name] = data
                except Exception as e:
                    logger.error(f"Failed to load variable '{var_name}': {e}")
                    raise
    else:
        # Fallback to netCDF4
        for var_name in variable_names:
            try:
                data = load_netcdf_wind_speed_netcdf4(
                    filepath, var_name, z_level, transpose
                )
                results[var_name] = data
            except Exception as e:
                logger.error(f"Failed to load variable '{var_name}': {e}")
                raise

    logger.info(f"Successfully loaded {len(results)} variables from {filepath}")
    return results


def load_netcdf_time_series(
    filepath: str,
    variable_name: str = "wind_speed",
    z_level: int = 1,
    time_range: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Load a time series of 2D wind data from 4D NetCDF files.

    This function handles 4D NetCDF files (time, z, y, x) and extracts
    a time series of 2D slices at a specified z-level.

    Parameters
    ----------
    filepath : str
        Path to the NetCDF file
    variable_name : str, optional
        Name of the wind speed variable (default: 'wind_speed')
    z_level : int, optional
        Z-level to extract (default: 1 for ground level)
    time_range : Tuple[int, int], optional
        Time range to extract as (start_time, end_time). If None, loads all time steps.

    Returns
    -------
    numpy.ndarray
        3D array with shape (time, height, width) containing the time series data

    Raises
    ------
    ImportError
        If xarray is not available (required for time series operations)
    FileNotFoundError
        If the NetCDF file doesn't exist
    ValueError
        If the data doesn't have a time dimension

    Examples
    --------
    >>> # Load full time series
    >>> time_series = load_netcdf_time_series('temporal_wind.nc', z_level=1)
    >>> print(time_series.shape)  # (24, 130, 130) for 24 time steps
    >>>
    >>> # Load specific time range
    >>> subset = load_netcdf_time_series('temporal_wind.nc', time_range=(0, 12))
    """
    if not XARRAY_AVAILABLE:
        raise ImportError(
            "xarray is required for time series operations. "
            "Install with: pip install xarray"
        )

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"NetCDF file not found: {filepath}")

    logger.debug(f"Loading time series from NetCDF file: {filepath}")

    try:
        with xr.open_dataset(filepath, decode_times=False) as ds:
            # Try to find the wind speed variable
            if variable_name not in ds.data_vars:
                alt_names = ["wind_speed", "Uped", "u", "velocity", "speed", "ws"]
                found_var = None

                for alt_name in alt_names:
                    if alt_name in ds.data_vars:
                        found_var = alt_name
                        break

                if found_var is None:
                    available_vars = list(ds.data_vars.keys())
                    raise KeyError(
                        f"Variable '{variable_name}' not found. Available variables: {available_vars}"
                    )

                variable_name = found_var
                logger.info(f"Using variable '{variable_name}' for wind speed data")

            data_var = ds[variable_name]

            if data_var.ndim < 3:
                raise ValueError(
                    "Time series loading requires at least 3D data (time, y, x) or 4D data (time, z, y, x)"
                )

            # Find time dimension
            time_dim_names = ["time", "t", "step"]
            time_dim = None

            for dim_name in data_var.dims:
                if dim_name.lower() in time_dim_names:
                    time_dim = dim_name
                    break

            if time_dim is None:
                raise ValueError("No time dimension found in the data")

            # Prepare selection dictionary
            selection = {}

            # Handle z-level selection for 4D data
            if data_var.ndim == 4:
                z_dim_names = [
                    "z",
                    "height",
                    "level",
                    "vertical",
                    "lev",
                    "alt",
                    "altitude",
                ]
                z_dim = None

                for dim_name in data_var.dims:
                    if dim_name.lower() in z_dim_names:
                        z_dim = dim_name
                        break

                if z_dim is not None:
                    if z_level < data_var.sizes[z_dim]:
                        selection[z_dim] = z_level
                    else:
                        max_z = data_var.sizes[z_dim] - 1
                        selection[z_dim] = max_z
                        logger.warning(
                            f"z_level {z_level} exceeds maximum {max_z}, using {max_z}"
                        )

            # Handle time range selection
            if time_range is not None:
                start_time, end_time = time_range
                max_time = data_var.sizes[time_dim]
                start_time = max(0, min(start_time, max_time - 1))
                end_time = max(start_time + 1, min(end_time, max_time))
                selection[time_dim] = slice(start_time, end_time)
                logger.info(f"Loading time range {start_time}:{end_time}")

            # Extract data
            if selection:
                result = data_var.isel(selection).values
            else:
                result = data_var.values

            # Convert to numpy array and ensure it's float32
            result = np.array(result, dtype=np.float32)

            # Handle NaN values
            nan_count = np.sum(np.isnan(result))
            if nan_count > 0:
                logger.info(f"Replacing {nan_count} NaN values with 0.0 in time series")
                result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

            logger.info(f"Loaded time series with shape: {result.shape}")
            return result

    except Exception as e:
        logger.error(f"Error loading time series from {filepath}: {e}")
        raise
