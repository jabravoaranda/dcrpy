from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dcrpy.dual_rpg import dual_rpg
from dcrpy.utils import parse_datetime


KA_LV0 = Path(r"tests\data\RAW\nebula_ka\2026\03\10\260310_020001_P00_ZEN.LV0")
W_LV0 = Path(r"tests\data\RAW\nebula_w\2026\03\10\260310_020001_P00_ZEN.LV0")
TIME_WINDOW_HOURS_FROM_START = (0, 4)
RANGE_WINDOW_METERS = (1000.0, 6000.0)
REQUESTED_RANGES_METERS = np.arange(1000.0, 6000.0 + 1.0, 500.0)
TEST_FILE_STEM = Path(__file__).stem


@pytest.fixture(scope="module")
def dual_radar() -> dual_rpg:
    return dual_rpg(KA_LV0, W_LV0)


@pytest.fixture()
def plot_output_root(tmp_path: Path, save_test_figures: bool) -> Path:
    if save_test_figures:
        output_dir = Path("tests/figures")
    else:
        output_dir = tmp_path / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _case_output_dir(root: Path, suffix: str) -> Path:
    output_dir = root / f"{TEST_FILE_STEM}-{suffix}"
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


def _write_selection_summary(output_dir: Path, **metadata: object) -> None:
    summary = output_dir / "selection.txt"
    lines = [f"{key}: {value}" for key, value in metadata.items()]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _constrained_dataset(dual_radar: dual_rpg) -> xr.Dataset:
    data = dual_radar.dataset
    time_values = pd.to_datetime(data.time.values)
    start_time = time_values[0] + pd.Timedelta(hours=TIME_WINDOW_HOURS_FROM_START[0])
    end_time = time_values[0] + pd.Timedelta(hours=TIME_WINDOW_HOURS_FROM_START[1])
    time_mask = (time_values >= start_time) & (time_values <= end_time)
    range_mask = (
        (data.range.values >= RANGE_WINDOW_METERS[0])
        & (data.range.values <= RANGE_WINDOW_METERS[1])
    )

    if not np.any(time_mask):
        raise AssertionError(
            "No data found within the requested time window from the start of the file."
        )
    if not np.any(range_mask):
        raise AssertionError(
            f"No data found between {RANGE_WINDOW_METERS[0]} and {RANGE_WINDOW_METERS[1]} m."
        )

    return data.isel(time=np.where(time_mask)[0], range=np.where(range_mask)[0])


def _target_ranges(dual_radar: dual_rpg) -> list[float]:
    data = _constrained_dataset(dual_radar)
    actual_ranges = np.unique(
        [
            float(data.sel(range=range_value, method="nearest").range.item())
            for range_value in REQUESTED_RANGES_METERS
        ]
    )
    return [float(value) for value in actual_ranges]


def _best_spectral_target_time(dual_radar: dual_rpg, ranges: list[float]) -> datetime:
    data = _constrained_dataset(dual_radar).sel(range=ranges, method="nearest")
    valid_mask = (
        np.isfinite(data["ka_doppler_spectrum_dB"]).any(dim="spectrum")
        & np.isfinite(data["w_doppler_spectrum_dB"]).any(dim="spectrum")
    )
    valid_counts = valid_mask.sum(dim="range").values
    time_index = int(np.argmax(valid_counts))
    return parse_datetime(data.time.values[time_index])


def _quicklook_time_limits(dual_radar: dual_rpg) -> tuple[datetime, datetime]:
    data = _constrained_dataset(dual_radar)
    return parse_datetime(data.time.values[0]), parse_datetime(data.time.values[-1])


def _count_labeled_lines(fig: Figure) -> int:
    return sum(
        1
        for line in fig.axes[0].lines
        if line.get_label() and not line.get_label().startswith("_")
    )


def test_plot_spectral_dfr_by_range_attenuation_check(
    dual_radar: dual_rpg,
    plot_output_root: Path,
):
    plot_output_dir = _case_output_dir(plot_output_root, "spectral_dfr_by_range")
    target_ranges = _target_ranges(dual_radar)
    target_time = _best_spectral_target_time(dual_radar, target_ranges)

    _write_selection_summary(
        plot_output_dir,
        test="plot_spectral_dfr_by_range_attenuation_check",
        target_time=target_time.isoformat(),
        requested_ranges_m=list(REQUESTED_RANGES_METERS.astype(float)),
        actual_ranges_m=target_ranges,
        time_window_hours_from_start=TIME_WINDOW_HOURS_FROM_START,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = dual_radar.plot_spectral_dfr_by_range(
        target_time,
        target_ranges,
        output_dir=plot_output_dir,
        savefig=True,
    )

    _assert_saved_figure(fig, filepath)
    assert len(fig.axes) == 1
    assert _count_labeled_lines(fig) >= 5
    plt.close(fig)


def test_plot_dfr_quicklook(
    dual_radar: dual_rpg,
    plot_output_root: Path,
):
    plot_output_dir = _case_output_dir(plot_output_root, "dfr_quicklook")
    time_limits = _quicklook_time_limits(dual_radar)

    _write_selection_summary(
        plot_output_dir,
        test="plot_dfr_quicklook",
        time_start=time_limits[0].isoformat(),
        time_end=time_limits[1].isoformat(),
        time_window_hours_from_start=TIME_WINDOW_HOURS_FROM_START,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = dual_radar.plot_dfr_quicklook(
        time_limits=time_limits,
        range_limits=RANGE_WINDOW_METERS,
        output_dir=plot_output_dir,
        savefig=True,
    )

    _assert_saved_figure(fig, filepath)
    assert len(fig.axes) >= 1
    plt.close(fig)
