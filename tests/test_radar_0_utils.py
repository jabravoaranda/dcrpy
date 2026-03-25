import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dcrpy.utils import (
    check_is_netcdf,
    histogram_intersection,
    parse_datetime,
    ppi_to_cartessian,
    rhi_to_cartessian,
)


CLOUDNET_NC = Path(r"tests\data\cloudnet\20211218_granada_rpg-fmcw-94.nc")


def test_check_is_netcdf_accepts_existing_path_and_string():
    assert check_is_netcdf(CLOUDNET_NC) == CLOUDNET_NC
    assert check_is_netcdf(CLOUDNET_NC.as_posix()) == CLOUDNET_NC


def test_check_is_netcdf_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        check_is_netcdf(Path("tests/data/cloudnet/missing-file.nc"))


def test_check_is_netcdf_raises_for_non_netcdf(tmp_path: Path):
    text_file = tmp_path / "not-a-netcdf.txt"
    text_file.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError):
        check_is_netcdf(text_file)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (dt.datetime(2024, 3, 13, 15, 0, 1), dt.datetime(2024, 3, 13, 15, 0, 1)),
        (dt.date(2024, 3, 13), dt.datetime(2024, 3, 13, 0, 0, 0)),
        ("2024-03-13T15:00:01", dt.datetime(2024, 3, 13, 15, 0, 1)),
        ("20240313T150001", dt.datetime(2024, 3, 13, 15, 0, 1)),
        (np.datetime64("2024-03-13T15:00:01"), dt.datetime(2024, 3, 13, 15, 0, 1)),
        (pd.Timestamp("2024-03-13T15:00:01"), dt.datetime(2024, 3, 13, 15, 0, 1)),
        (
            pd.DatetimeIndex(["2024-03-13T15:00:01"]),
            dt.datetime(2024, 3, 13, 15, 0, 1),
        ),
    ],
)
def test_parse_datetime_supported_inputs(value, expected):
    assert parse_datetime(value) == expected


def test_parse_datetime_raises_for_invalid_type():
    with pytest.raises(ValueError):
        parse_datetime(12345)


def test_ppi_to_cartessian_returns_expected_coordinates():
    ranges = np.array([0.0, 10.0, 20.0])
    azimuth = np.array([0.0, 90.0, 180.0, 270.0])
    elevation = np.full(azimuth.shape, 0.0)

    x, y = ppi_to_cartessian(ranges, azimuth, elevation)

    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    np.testing.assert_allclose(x[1], [0.0, 10.0, 0.0, -10.0], atol=1e-6)
    np.testing.assert_allclose(y[1], [10.0, 0.0, -10.0, 0.0], atol=1e-6)


def test_ppi_to_cartessian_accepts_xarray_inputs():
    ranges = xr.DataArray(np.array([0.0, 10.0, 20.0]), dims=("range",))
    azimuth = xr.DataArray(np.array([0.0, 90.0]), dims=("azimuth",))
    elevation = xr.DataArray(np.full(azimuth.shape, 45.0), dims=("azimuth",))

    x, y = ppi_to_cartessian(ranges, azimuth, elevation)

    assert x.shape == (3, 2)
    assert y.shape == (3, 2)


def test_ppi_to_cartessian_raises_for_non_constant_elevation():
    ranges = np.array([0.0, 10.0, 20.0])
    azimuth = np.array([0.0, 90.0, 180.0])
    elevation = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="Elevation angle is not constant"):
        ppi_to_cartessian(ranges, azimuth, elevation)


def test_rhi_to_cartessian_returns_expected_coordinates():
    ranges = np.array([0.0, 10.0, 20.0])
    elevation = np.array([0.0, 30.0, 90.0])
    azimuth = np.full(elevation.shape, 180.0)

    x, y = rhi_to_cartessian(ranges, azimuth, elevation)

    assert x.shape == (3, 3)
    assert y.shape == (3, 3)
    np.testing.assert_allclose(x[1], [10.0, 10.0 * np.cos(np.deg2rad(30.0)), 0.0], atol=1e-6)
    np.testing.assert_allclose(y[1], [0.0, 10.0 * np.sin(np.deg2rad(30.0)), 10.0], atol=1e-6)


def test_rhi_to_cartessian_accepts_xarray_inputs():
    ranges = xr.DataArray(np.array([0.0, 10.0, 20.0]), dims=("range",))
    elevation = xr.DataArray(np.array([10.0, 20.0, 30.0]), dims=("elevation",))
    azimuth = xr.DataArray(np.full(elevation.shape, 180.0), dims=("elevation",))

    x, y = rhi_to_cartessian(ranges, azimuth, elevation)

    assert x.shape == (3, 3)
    assert y.shape == (3, 3)


def test_rhi_to_cartessian_raises_for_non_constant_azimuth():
    ranges = np.array([0.0, 10.0, 20.0])
    elevation = np.array([10.0, 20.0, 30.0])
    azimuth = np.array([180.0, 181.0, 182.0])

    with pytest.raises(ValueError, match="Azimuth angle is not constant"):
        rhi_to_cartessian(ranges, azimuth, elevation)


def test_histogram_intersection_is_symmetric_and_weighted_by_bin_width():
    histogram1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    histogram2 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    bins = np.arange(0.0, 6.0, 1.0)

    intersection = histogram_intersection(histogram1, histogram2, bins)

    assert intersection == 9.0
    assert intersection == histogram_intersection(histogram2, histogram1, bins)


def test_histogram_intersection_handles_non_uniform_bins():
    histogram1 = np.array([2.0, 1.0, 4.0])
    histogram2 = np.array([1.0, 3.0, 2.0])
    bins = np.array([0.0, 1.0, 3.0, 6.0])

    intersection = histogram_intersection(histogram1, histogram2, bins)

    expected = 1.0 * 1.0 + 2.0 * 1.0 + 3.0 * 2.0
    assert intersection == expected
