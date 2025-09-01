"""
Regression tests for NetCDF loader functionality with corrected (x, y, z) dimensions.

These tests ensure that:
1. z_level=1 extracts meaningful wind speed data (non-zero values)
2. z_level=0 contains surface data (typically zeros)
3. The corrected NetCDF file structure is properly handled
4. Data extraction produces correct 2D arrays with expected properties

Requirements:
- NetCDF files with corrected (x, y, z) dimensions in raw_data/wind3D/idealized/
- xarray and/or netCDF4 libraries installed
"""

import pytest
import numpy as np
import pathlib
from typing import Optional


def find_sample_netcdf_file() -> Optional[pathlib.Path]:
    """Find a sample NetCDF file for testing."""
    netcdf_dir = pathlib.Path("raw_data/wind3D/idealized")
    
    if not netcdf_dir.exists():
        return None
        
    # Look for any .nc file
    for nc_file in netcdf_dir.glob("*.nc"):
        if nc_file.is_file():
            return nc_file
    
    return None


@pytest.fixture(scope="module")
def sample_netcdf_file():
    """Fixture to provide a sample NetCDF file for testing."""
    sample_file = find_sample_netcdf_file()
    
    if sample_file is None:
        pytest.skip("No NetCDF files found in raw_data/wind3D/idealized/")
    
    return sample_file


def test_netcdf_availability():
    """Test that NetCDF loading libraries are available."""
    try:
        from feilian.netcdf_loader import is_netcdf_available
        assert is_netcdf_available(), "NetCDF support should be available"
    except ImportError:
        pytest.fail("feilian.netcdf_loader module not available")


def test_z_level_extraction(sample_netcdf_file):
    """
    Test that z_level extraction works correctly with corrected NetCDF structure.
    
    This test verifies:
    - z=1 contains meaningful wind speed data (non-zero values)
    - z=0 contains surface data (typically all zeros)
    - Both extractions produce valid 2D float32 arrays
    """
    from feilian.netcdf_loader import load_netcdf_wind_speed
    
    # Test z=1 (ground level with wind speeds)
    z1_data = load_netcdf_wind_speed(str(sample_netcdf_file), z_level=1)
    
    # Basic structure validation
    assert z1_data.ndim == 2, f"z=1 data should be 2D, got {z1_data.ndim}D"
    assert z1_data.dtype == np.float32, f"z=1 data should be float32, got {z1_data.dtype}"
    assert z1_data.size > 0, "z=1 data should not be empty"
    
    # Content validation - should have meaningful wind speeds
    nonzero_count = np.count_nonzero(z1_data)
    assert nonzero_count > 0, f"z=1 should contain non-zero wind speeds, got {nonzero_count}/{z1_data.size} non-zero values"
    
    # Test z=0 (surface level, typically zeros)
    z0_data = load_netcdf_wind_speed(str(sample_netcdf_file), z_level=0)
    
    # Basic structure validation
    assert z0_data.ndim == 2, f"z=0 data should be 2D, got {z0_data.ndim}D"
    assert z0_data.dtype == np.float32, f"z=0 data should be float32, got {z0_data.dtype}"
    assert z0_data.shape == z1_data.shape, f"z=0 and z=1 should have same shape, got {z0_data.shape} vs {z1_data.shape}"
    
    # Content validation - surface level should typically be zeros
    z0_nonzero_count = np.count_nonzero(z0_data)
    assert z0_nonzero_count == 0, f"z=0 slice should be all zeros, got {z0_nonzero_count}/{z0_data.size} non-zero values"
    
    # Statistical validation
    assert z1_data.min() >= 0.0, f"Wind speeds should be non-negative, got min={z1_data.min()}"
    assert z1_data.max() > 0.0, f"Should have some positive wind speeds, got max={z1_data.max()}"
    
    print(f"✓ z=1 data shape: {z1_data.shape}, range: [{z1_data.min():.6f}, {z1_data.max():.6f}]")
    print(f"✓ z=0 data shape: {z0_data.shape}, all zeros: {z0_nonzero_count == 0}")


def test_universal_loader(sample_netcdf_file):
    """Test that the universal load_wind_data function works with corrected NetCDF files."""
    from feilian.netcdf_loader import load_wind_data
    
    # Test loading with explicit z_level=1
    data = load_wind_data(str(sample_netcdf_file), netcdf_variable='wind_speed', z_level=1)
    
    assert data.ndim == 2, "Universal loader should return 2D array"
    assert data.dtype == np.float32, "Universal loader should return float32"
    assert np.count_nonzero(data) > 0, "Should contain non-zero wind speeds"
    
    print(f"✓ Universal loader: shape {data.shape}, {np.count_nonzero(data)}/{data.size} non-zero values")


def test_netcdf_info_extraction(sample_netcdf_file):
    """Test that NetCDF file info extraction works with corrected structure."""
    from feilian.netcdf_loader import get_netcdf_info
    
    info = get_netcdf_info(str(sample_netcdf_file))
    
    # Check that we have the expected structure
    assert 'variables' in info, "Info should contain variables"
    assert 'dimensions' in info, "Info should contain dimensions"
    
    # Check for wind_speed variable
    assert 'wind_speed' in info['variables'], "Should have wind_speed variable"
    
    wind_speed_var = info['variables']['wind_speed']
    
    # Verify the corrected 3D structure (x, y, z)
    assert len(wind_speed_var['shape']) == 3, f"Wind speed should be 3D, got shape {wind_speed_var['shape']}"
    assert wind_speed_var['dimensions'] == ('x', 'y', 'z'), f"Should have (x, y, z) dimensions, got {wind_speed_var['dimensions']}"
    
    # Check dimension sizes are reasonable
    x_size, y_size, z_size = wind_speed_var['shape']
    assert x_size > 0 and y_size > 0 and z_size > 0, f"All dimensions should be positive: {wind_speed_var['shape']}"
    assert z_size >= 2, f"Should have at least 2 z-levels (surface + ground), got {z_size}"
    
    print(f"✓ NetCDF structure: {wind_speed_var['dimensions']} with shape {wind_speed_var['shape']}")


def test_variable_name_fallback(sample_netcdf_file):
    """Test that alternative variable names are handled correctly."""
    from feilian.netcdf_loader import load_netcdf_wind_speed
    
    # Test with non-existent variable name - should fall back to 'wind_speed'
    try:
        data = load_netcdf_wind_speed(str(sample_netcdf_file), variable_name='nonexistent_var', z_level=1)
        # If this succeeds, it means the fallback mechanism worked
        assert data.ndim == 2, "Fallback should still return valid 2D data"
        print("✓ Variable name fallback mechanism working")
    except KeyError:
        # This is also acceptable - means the file only has 'wind_speed'
        print("✓ Strict variable name checking (no fallback needed)")


def test_nan_handling(sample_netcdf_file):
    """Test that NaN values are properly handled."""
    from feilian.netcdf_loader import load_netcdf_wind_speed
    
    data = load_netcdf_wind_speed(str(sample_netcdf_file), z_level=1)
    
    # Check that there are no NaN values (should be replaced with zeros)
    nan_count = np.sum(np.isnan(data))
    assert nan_count == 0, f"Should have no NaN values after loading, got {nan_count}"
    
    # Check for infinite values
    inf_count = np.sum(np.isinf(data))
    assert inf_count == 0, f"Should have no infinite values, got {inf_count}"
    
    print(f"✓ NaN/Inf handling: {nan_count} NaNs, {inf_count} Infs after processing")


if __name__ == "__main__":
    # Allow running tests directly for development
    sample_file = find_sample_netcdf_file()
    
    if sample_file is None:
        print("❌ No NetCDF files found in raw_data/wind3D/idealized/")
        print("Please ensure corrected NetCDF files are available for testing.")
        exit(1)
    
    print(f"Running tests with sample file: {sample_file}")
    
    try:
        # Run basic tests
        test_netcdf_availability()
        test_z_level_extraction(sample_file)
        test_universal_loader(sample_file)
        test_netcdf_info_extraction(sample_file)
        test_variable_name_fallback(sample_file)
        test_nan_handling(sample_file)
        
        print("\n✅ All NetCDF loader tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
