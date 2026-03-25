from datetime import datetime
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from dcrpy.retrieve.moments import ze_from_spectrum
from dcrpy.retrieve.retrieve import retrieve_ZDR, retrieve_dBZe, retrieve_spectral_ZDR
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
TEST_FILE_STEM = Path(__file__).stem


@pytest.fixture(scope="module")
def radar() -> rpg:
    return rpg(ZEN)


@pytest.fixture()
def plot_output_root(tmp_path: Path, save_test_figures: bool) -> Path:
    if save_test_figures:
        output_dir = Path("tests/figures")
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


def _write_selection_summary(output_dir: Path, **metadata: object) -> None:
    summary = output_dir / "selection.txt"
    lines = [f"{key}: {value}" for key, value in metadata.items()]
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_output_dir(root: Path, suffix: str) -> Path:
    output_dir = root / f"{TEST_FILE_STEM}-{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def _sample_time_and_range(radar: rpg) -> tuple[datetime, float]:
    data = _constrained_dataset(radar)
    valid_mask = np.isfinite(data["doppler_spectrum_h"]).any(dim="spectrum")
    valid_ranges_per_time = valid_mask.sum(dim="range").values
    time_index = int(np.argmax(valid_ranges_per_time))

    valid_range_indices = np.where(valid_mask.isel(time=time_index).values)[0]
    if valid_range_indices.size == 0:
        valid_range_indices = np.arange(data.sizes["range"])

    range_index = int(valid_range_indices[min(1, valid_range_indices.size - 1)])
    target_time = parse_datetime(data.time.values[time_index])
    target_range = float(data.range.values[range_index])
    return target_time, target_range


def _spectral_line(
    dataset: xr.Dataset,
    variable: str,
    target_time: datetime,
    target_range: float,
) -> xr.DataArray:
    chirp_number = int(
        dataset["chirp_number"].sel(range=target_range, method="nearest").item()
    )
    selected = dataset.sel(time=target_time, range=target_range, method="nearest")[variable]
    velocity = dataset["velocity_vectors"].sel(chirp=chirp_number)
    return selected.assign_coords(spectrum=velocity)


@pytest.mark.parametrize("variable", ["dBZe", "ZDR"])
def test_plot_retrieve_time_height_products(
    radar: rpg, plot_output_root: Path, variable: str
):
    plot_output_dir = _case_output_dir(plot_output_root, f"time_height_{variable}")
    data = _constrained_dataset(radar)
    if variable == "dBZe":
        product = retrieve_dBZe(ze_from_spectrum(data), radar.band)
    else:
        product = retrieve_ZDR(data)

    product = product.assign_coords(range=data["range"] / 1e3)
    _write_selection_summary(
        plot_output_dir,
        test="plot_retrieve_time_height_products",
        variable=variable,
        time_start=parse_datetime(data.time.values[0]).isoformat(),
        time_end=parse_datetime(data.time.values[-1]).isoformat(),
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    product.plot(x="time", y="range", ax=ax, cmap="viridis")  # type: ignore[arg-type]
    ax.set_ylabel("Range, [km]")
    ax.set_title(f"{variable} time-height")
    fig.tight_layout()

    filepath = plot_output_dir / f"{ZEN.stem}_{variable}_time_height.png"
    fig.savefig(filepath, dpi=300)

    _assert_saved_figure(fig, filepath)
    plt.close(fig)


def test_plot_retrieve_overlay_spectra(radar: rpg, plot_output_root: Path):
    plot_output_dir = _case_output_dir(plot_output_root, "overlay_spectra")
    data = _constrained_dataset(radar)
    target_time, target_range = _sample_time_and_range(radar)
    total_dB = retrieve_dBZe(
        _spectral_line(data, "doppler_spectrum", target_time, target_range), radar.band
    )
    h_dB = retrieve_dBZe(
        _spectral_line(data, "doppler_spectrum_h", target_time, target_range), radar.band
    )
    v_dB = retrieve_dBZe(
        _spectral_line(data, "doppler_spectrum_v", target_time, target_range), radar.band
    )

    _write_selection_summary(
        plot_output_dir,
        test="plot_retrieve_overlay_spectra",
        target_time=target_time.isoformat(),
        target_range_m=target_range,
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    total_dB.plot(ax=ax, color="black", label="total")  # type: ignore[arg-type]
    h_dB.plot(ax=ax, color="tab:blue", label="horizontal")  # type: ignore[arg-type]
    v_dB.plot(ax=ax, color="tab:red", label="vertical")  # type: ignore[arg-type]
    ax.set_xlabel("Doppler velocity, [m/s]")
    ax.set_ylabel("Power density, [dB]")
    ax.set_title(f"Retrieved spectra at {target_time:%Y-%m-%d %H:%M:%S}, {target_range:.1f} m")
    ax.axvline(0.0, color="gray", linestyle="--")
    ax.legend()
    fig.tight_layout()

    filepath = plot_output_dir / f"{ZEN.stem}_retrieve_overlay_spectra.png"
    fig.savefig(filepath, dpi=300)

    _assert_saved_figure(fig, filepath)
    assert len(ax.lines) >= 4
    plt.close(fig)


def test_plot_retrieve_spectral_zdr_line(radar: rpg, plot_output_root: Path):
    plot_output_dir = _case_output_dir(plot_output_root, "spec_zdr")
    data = _constrained_dataset(radar)
    target_time, target_range = _sample_time_and_range(radar)
    spectral_zdr = retrieve_spectral_ZDR(data)
    spectral_zdr_line = _spectral_line(
        data.assign(spec_ZDR=spectral_zdr),
        "spec_ZDR",
        target_time,
        target_range,
    )

    _write_selection_summary(
        plot_output_dir,
        test="plot_retrieve_spectral_zdr_line",
        target_time=target_time.isoformat(),
        target_range_m=target_range,
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    spectral_zdr_line.plot(ax=ax, color="tab:purple", label="spec_ZDR")  # type: ignore[arg-type]
    ax.set_xlabel("Doppler velocity, [m/s]")
    ax.set_ylabel("Spectral ZDR, [dB]")
    ax.set_title(
        f"Retrieved spectral ZDR at {target_time:%Y-%m-%d %H:%M:%S}, {target_range:.1f} m"
    )
    ax.axvline(0.0, color="gray", linestyle="--")
    ax.legend()
    fig.tight_layout()

    filepath = plot_output_dir / f"{ZEN.stem}_retrieve_spec_zdr.png"
    fig.savefig(filepath, dpi=300)

    _assert_saved_figure(fig, filepath)
    assert len(ax.lines) >= 2
    plt.close(fig)


def test_plot_retrieve_vertical_2d_spectrum(radar: rpg, plot_output_root: Path):
    plot_output_dir = _case_output_dir(plot_output_root, "vertical_2d_spectrum")
    target_time, _ = _sample_time_and_range(radar)
    _write_selection_summary(
        plot_output_dir,
        test="plot_retrieve_vertical_2d_spectrum",
        variable="doppler_spectrum_v",
        target_time=target_time.isoformat(),
        time_window_hours=TIME_WINDOW_HOURS,
        range_window_m=RANGE_WINDOW_METERS,
    )

    fig, filepath = radar.plot_2D_spectrum(
        target_time=target_time,
        range_limits=RANGE_WINDOW_METERS,
        variable_to_plot="doppler_spectrum_v",
        output_dir=plot_output_dir,
        power_spectrum_limits=(-60, 20),
    )

    _assert_saved_figure(fig, filepath)
    plot_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
    assert any(ax.images for ax in plot_axes)
    plt.close(fig)
