
from pathlib import Path
from pdb import set_trace
import numpy as np
from gfatpy.radar.utils import check_is_netcdf, histogram_intersection, ppi_to_cartessian, rhi_to_cartessian

ZEN_NC = Path(r"tests\datos\PRODUCTS\nebula_ka\2024\03\13\240313_150001_P00_ZEN.LV1.nc")

def test_check_path(radar_files):
    path = check_is_netcdf(ZEN_NC)
    assert path.exists()

def test_ppi_to_cartessian():

    ranges = np.arange(0., 100., 10.)
    azimuth = np.arange(0., 360., 36.)
    elevation = np.full(azimuth.shape, 45.)

    x, y = ppi_to_cartessian(ranges, azimuth, elevation)

    assert x.shape == (10, 10)
    assert y.shape == (10, 10)

def test_rhi_to_cartessian():

    ranges = np.arange(0, 100, 10)    
    elevation = np.arange(30, 90, 10)
    azimuth = elevation = np.full(elevation.shape, 180.)

    x, y = rhi_to_cartessian(ranges, azimuth, elevation)
    assert x.shape == (10, 6)
    assert y.shape == (10, 6)

def test_histogram_intersection():
    histrogram1 = np.array([1, 2, 3, 4, 5])
    histrogram2 = np.array([5, 4, 3, 2, 1])
    bins = np.arange(0, 6, 1)
    intersection = histogram_intersection(histrogram1, histrogram2, bins)
    assert intersection == 9.0