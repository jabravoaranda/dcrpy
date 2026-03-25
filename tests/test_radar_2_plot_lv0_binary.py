from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dcrpy.rpg_binary import rpg
from dcrpy.utils import parse_datetime


LV0_CANDIDATES = (
    Path(r"tests\data\RAW\nebula_w\2026\03\10\260310_020001_P00_ZEN.LV0"),
    Path(r"tests\data\RAW\nebula_w\2024\03\13\240313_150001_P00_ZEN.LV0"),
)
TIME_WINDOW_HOURS = (0, 4)
RANGE_WINDOW_METERS = (1000.0, 6000.0)


def _select_lv0_file() -> Path:
    for candidate in LV0_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No suitable LV0 test file found. Checked: "
        + ", ".join(path.as_posix() for path in LV0_CANDIDATES)
    )


ZEN = _select_lv0_file()


@pytest.fixture(scope="module")
def radar() -> rpg:
    return rpg(ZEN)


@pytest.fixture()
def plot_output_dir(
    tmp_path: Path, request: pytest.FixtureRequest, save_test_figures: bool
) -> Path:
    if save_test_figures:
        safe_name = request.node.name
        for old, new in (
            ("[", "_"),
            ("]", ""),
            (":", "_"),
            ("/", "_"),
            ("\\", "_"),
            (" ", "_"),
        ):
            safe_name = safe_name.replace(old, new)
        output_dir = Path("tests/figures") / safe_name
    else:
        output_dir = tmp_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _assert_saved_figure(fig: Figure, filepath: Path | None) -> None:
    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()
    assert filepath.suffix.lower() == ".png"
    assert filepath.stat().st_size > 0

    image = mpimg.imread(filepath)
    assert image.ndim in (2, 3)
    assert np.isfinite(image).all()
    assert float(np.ptp(image.astype(np.float32))) > 0.0


def _count_labeled_lines(fig: Figure) -> int:
    return sum(
        1
        for line in fig.axes[0].lines
        if line.get_label() and not line.get_label().startswith("_")
    )


def _write_selection_summary(output_dir: Path, **metadata: object) -> None:
    summary = output_dir / "selection.txt"
    lines = [f"{key}: {value}" for key, value in metadata.items()]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _constrained_dataset(radar: rpg) -> xr.Dataset:
    data = radar.dataset
    time_values = pd.to_datetime(data.time.values)
    start_time = time_values[0].normalize() + pd.Timedelta(hours=TIME_WINDOW_HOURS[0])
    end_time = time_values[0].normalize() + pd.Timedelta(hours=TIME_WINDOW_HOURS[1])

    time_mask = (time_values >= start_time) & (time_values <= end_time)
    range_mask = (
        (data.range.values >= RANGE_WINDOW_METERS[0])
        & (data.range.values <= RANGE_WINDOW_METERS[1])
    )

    if not np.any(time_mask):
        raise AssertionError(
            f"No data found between {TIME_WINDOW_HOURS[0]}h and {TIME_WINDOW_HOURS[1]}h."
        )
    if not np.any(range_mask):
        raise AssertionError(
            f"No data found between {RANGE_WINDOW_METERS[0]} m and "
            f"{RANGE_WINDOW_METERS[1]} m."
        )

    return data.isel(time=np.where(time_mask)[0], range=np.where(range_mask)[0])


def _sample_time_and_ranges(radar: rpg) -> tuple[datetime, list[float]]:
    data = _constrained_dataset(radar)
    valid_mask = np.isfinite(data["doppler_spectrum_h"]).any(dim="spectrum")
    valid_ranges_per_time = valid_mask.sum(dim="range").values
    time_index = int(np.argmax(valid_ranges_per_time))

    valid_range_indices = np.where(valid_mask.isel(time=time_index).values)[0]
    if valid_range_indices.size == 0:
        valid_range_indices = np.arange(data.sizes["range"])

    sample_positions = np.linspace(
        0,
        valid_range_indices.size - 1,
        num=min(4, valid_range_indices.size),
        dtype=int,
    )
    range_indices = valid_range_indices[sample_positions]
    range_values = [float(data.range.values[idx]) for idx in range_indices]
    return parse_datetime(data.time.values[time_index]), range_values


def _sample_time_slice_for_range(radar: rpg) -> tuple[float, tuple[datetime, datetime]]:
    data = _constrained_dataset(radar)
    valid_mask = np.isfinite(data["doppler_spectrum"]).any(dim="spectrum")
    valid_times_per_range = valid_mask.sum(dim="time").values
    range_index = int(np.argmax(valid_times_per_range))
    target_range = float(data.range.values[range_index])

    valid_time_indices = np.where(valid_mask.isel(range=range_index).values)[0]
    if valid_time_indices.size == 0:
        valid_time_indices = np.arange(data.sizes["time"])

    start_index = int(valid_time_indices[0])
    end_index = int(valid_time_indices[min(3, valid_time_indices.size - 1)])
    return target_range, (
        parse_datetime(data.time.values[start_index]),
        parse_datetime(data.time.values[end_index]),
    )


def test_plot_spectrum_multi_variable_overlay(radar: rpg, plot_output_dir: Path):
    target_time, range_values = _sample_time_and_ranges(radar)
    target_range = range_values[min(1, len(range_values) - 1)]
    _write_selection_summary(
        plot_output_dir,
        test="plot_spectrum_multi_variable_overlay",
        variables="doppler_spectrum,doppler_spectrum_h,doppler_spectrum_v",
        target_time=target_time.isoformat(),
        target_range_m=target_range,
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = radar.plot_spectrum(
        target_range=target_range,
        target_time=target_time,
        variable_to_plot=[
            "doppler_spectrum",
            "doppler_spectrum_h",
            "doppler_spectrum_v",
        ],
        output_dir=plot_output_dir,
    )

    _assert_saved_figure(fig, filepath)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Power density, [dB]"
    assert _count_labeled_lines(fig) >= 3
    plt.close(fig)


def test_plot_spectra_by_range(radar: rpg, plot_output_dir: Path):
    target_time, range_values = _sample_time_and_ranges(radar)
    if len(range_values) < 2:
        raise AssertionError("Need at least two valid range gates for spectra-by-range.")
    _write_selection_summary(
        plot_output_dir,
        test="plot_spectra_by_range",
        variable="doppler_spectrum_h",
        target_time=target_time.isoformat(),
        target_ranges_m=",".join(f"{range_value:.1f}" for range_value in range_values),
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = radar.plot_spectra_by_range(
        target_time=target_time,
        range_slice=range_values,
        output_dir=plot_output_dir,
        variable_to_plot="doppler_spectrum_h",
    )

    _assert_saved_figure(fig, filepath)
    ax = fig.axes[0]
    assert "Range:" in ax.get_title()
    assert _count_labeled_lines(fig) >= 2
    plt.close(fig)


@pytest.mark.parametrize(
    "variable_to_plot",
    ["doppler_spectrum", "doppler_spectrum_h"],
)
def test_plot_2d_spectrum(radar: rpg, plot_output_dir: Path, variable_to_plot: str):
    target_time, range_values = _sample_time_and_ranges(radar)
    range_slice = (range_values[0], range_values[-1])
    _write_selection_summary(
        plot_output_dir,
        test="plot_2d_spectrum",
        variable=variable_to_plot,
        target_time=target_time.isoformat(),
        range_limits_m=range_slice,
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = radar.plot_2D_spectrum(
        variable_to_plot=variable_to_plot,
        target_time=target_time,
        range_limits=range_slice,
        output_dir=plot_output_dir,
        power_spectrum_limits=(-60, 20),
    )

    _assert_saved_figure(fig, filepath)
    plot_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert len(plot_axes) >= 1
    assert any(ax.images for ax in plot_axes)
    plt.close(fig)


def test_plot_spectra_by_time(radar: rpg, plot_output_dir: Path):
    target_range, time_slice = _sample_time_slice_for_range(radar)
    _write_selection_summary(
        plot_output_dir,
        test="plot_spectra_by_time",
        variable="doppler_spectrum",
        target_range_m=target_range,
        time_slice=f"{time_slice[0].isoformat()} -> {time_slice[1].isoformat()}",
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = radar.plot_spectra_by_time(
        target_range=target_range,
        time_slice=time_slice,
        output_dir=plot_output_dir,
        variable_to_plot="doppler_spectrum",
    )

    _assert_saved_figure(fig, filepath)
    ax = fig.axes[0]
    assert "Period:" in ax.get_title()
    assert _count_labeled_lines(fig) >= 2
    plt.close(fig)


@pytest.mark.parametrize("variable", ["dBZe", "sZDRmax"])
def test_plot_profile(radar: rpg, plot_output_dir: Path, variable: str):
    data = _constrained_dataset(radar)
    time_index = min(10, data.sizes["time"] - 1)
    time_end_index = min(time_index + 2, data.sizes["time"] - 1)
    times = pd.date_range(
        start=parse_datetime(data.time.values[time_index]),
        end=parse_datetime(data.time.values[time_end_index]),
        freq="10min",
    ).to_pydatetime().tolist()
    range_limits = (float(data.range.values[0]), float(data.range.values[-1]))
    _write_selection_summary(
        plot_output_dir,
        test="plot_profile",
        variable=variable,
        target_times=",".join(time_value.isoformat() for time_value in times),
        range_limits_m=range_limits,
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = radar.plot_profile(
        target_times=times,
        range_limits=range_limits,
        variable=variable,
        output_dir=plot_output_dir,
        savefig=True,
        ncol=1,
    )

    _assert_saved_figure(fig, filepath)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Range, [km]"
    assert variable in ax.get_xlabel() or ax.get_xlabel()
    assert _count_labeled_lines(fig) >= 1
    plt.close(fig)
