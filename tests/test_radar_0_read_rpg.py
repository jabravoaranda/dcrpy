import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pytest
from rpgpy import read_rpg


RAW_CASES = [
    pytest.param(
        Path(r"tests\data\RAW\nebula_w\2024\03\13\240313_150001_P00_ZEN.LV0"),
        id="nebula-w-lv0",
    ),
    pytest.param(
        Path(r"tests\data\RAW\nebula_ka\2024\03\13\240313_150001_P00_ZEN.LV0"),
        id="nebula-ka-lv0",
    ),
]

REQUIRED_HEADER_KEYS = {
    "FileCode",
    "Freq",
    "MaxVel",
    "RAlts",
    "RangeMax",
    "RngOffs",
    "SWVersion",
    "SpecN",
    "velocity_vectors",
}
REQUIRED_DATA_KEYS = {"Time", "TotSpec", "HSpec", "ReVHSpec"}


def _assert_read_rpg_output(header: dict, data: dict) -> None:
    assert isinstance(header, dict)
    assert isinstance(data, dict)

    missing_header_keys = REQUIRED_HEADER_KEYS.difference(header)
    missing_data_keys = REQUIRED_DATA_KEYS.difference(data)
    assert not missing_header_keys, f"Missing header keys: {sorted(missing_header_keys)}"
    assert not missing_data_keys, f"Missing data keys: {sorted(missing_data_keys)}"

    velocity_vectors = np.asarray(header["velocity_vectors"])
    assert velocity_vectors.ndim == 2
    n_chirps, n_spectrum = velocity_vectors.shape

    time_values = np.asarray(data["Time"])
    range_values = np.asarray(header["RAlts"])
    assert time_values.ndim == 1
    assert range_values.ndim == 1

    n_time = time_values.shape[0]
    n_range = range_values.shape[0]
    expected_spectral_shape = (n_time, n_range, n_spectrum)

    for key in ("TotSpec", "HSpec", "ReVHSpec"):
        assert np.asarray(data[key]).shape == expected_spectral_shape

    for key in ("MaxVel", "RangeMax", "RngOffs", "SpecN"):
        assert np.asarray(header[key]).shape == (n_chirps,)

    chirp_start_indices = np.asarray(header["RngOffs"], dtype=int)
    assert np.all(chirp_start_indices >= 0)
    assert np.all(chirp_start_indices < n_range)
    assert np.all(np.diff(chirp_start_indices) >= 0)


@pytest.mark.parametrize("lv0_file", RAW_CASES)
def test_read_rpg_returns_expected_structure(lv0_file: Path):
    assert lv0_file.exists(), f"Missing test file: {lv0_file}"

    header, data = read_rpg(lv0_file)

    _assert_read_rpg_output(header, data)


def test_read_rpg_performance_smoke(record_property):
    if os.getenv("DCRPY_RUN_PERF") != "1":
        pytest.skip("Set DCRPY_RUN_PERF=1 to run read-performance smoke tests.")

    lv0_file = Path(r"tests\data\RAW\nebula_w\2024\03\13\240313_150001_P00_ZEN.LV0")
    repeats = int(os.getenv("DCRPY_READ_RPG_REPEATS", "3"))
    max_median_seconds = os.getenv("DCRPY_READ_RPG_MAX_MEDIAN_SECONDS")
    assert lv0_file.exists(), f"Missing test file: {lv0_file}"
    assert repeats >= 1

    durations = []
    for _ in range(repeats):
        start = perf_counter()
        header, data = read_rpg(lv0_file)
        durations.append(perf_counter() - start)
        _assert_read_rpg_output(header, data)

    median_seconds = median(durations)
    record_property("read_rpg_file", lv0_file.as_posix())
    record_property("read_rpg_repeats", repeats)
    record_property("read_rpg_median_seconds", median_seconds)
    record_property("read_rpg_min_seconds", min(durations))
    record_property("read_rpg_max_seconds", max(durations))

    if max_median_seconds is not None:
        assert median_seconds <= float(max_median_seconds), (
            f"Median read_rpg time {median_seconds:.3f}s exceeded "
            f"{float(max_median_seconds):.3f}s for {lv0_file.name}"
        )
