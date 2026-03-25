"""Combine, align, and plot dual-frequency RPG binary radar datasets.

This module defines :class:`dual_rpg`, a composition-based wrapper around two
single-frequency :class:`dcrpy.rpg_binary.rpg` objects. It aligns Ka- and
W-band datasets onto a common time/range grid, exposes a cached combined
dataset, and provides plotting helpers for dual-frequency reflectivity
products such as ``DFR_Ka_W`` and spectral DFR by range.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import xarray as xr

from dcrpy.plotting.utils import color_list
from dcrpy.rpg_binary import rpg
from dcrpy.retrieve.retrieve import retrieve_dBZe
from dcrpy.utils import parse_datetime


class dual_rpg:
    """Container for paired Ka- and W-band RPG binary datasets.

    Parameters
    ----------
    first, second : dcrpy.rpg_binary.rpg or str or pathlib.Path
        Two single-band radar inputs. They may be existing :class:`rpg`
        instances or paths from which :class:`rpg` objects will be built.
    reference_band : {"Ka", "W"}, default: "W"
        Band whose time/range grid becomes the reference grid for the combined
        dataset.
    time_tolerance : numpy.timedelta64, datetime.timedelta, or str, default: numpy.timedelta64(5, "s")
        Tolerance used for nearest-neighbor temporal alignment of the secondary
        radar onto the reference grid.
    range_tolerance : float, default: 60.0
        Range-alignment tolerance in meters.

    Notes
    -----
    The combined dataset stores prefixed single-band variables such as
    ``ka_dBZe`` and ``w_dBZe`` and adds ``DFR_Ka_W = ka_dBZe - w_dBZe``.
    Spectral DFR is stored directly only when Ka and W share the same Doppler
    velocity grid; otherwise it is computed on demand by interpolation onto the
    reference-band velocity grid.
    """

    _mergeable_dims = frozenset(
        {
            frozenset({"time", "range"}),
            frozenset({"time", "range", "spectrum"}),
            frozenset({"range"}),
            frozenset({"chirp"}),
            frozenset({"chirp", "spectrum"}),
        }
    )

    def __init__(
        self,
        first: rpg | str | Path,
        second: rpg | str | Path,
        *,
        reference_band: str = "W",
        time_tolerance: np.timedelta64 | timedelta | str = np.timedelta64(5, "s"),
        range_tolerance: float = 60.0,
    ):
        """Initialize the dual-frequency wrapper."""
        self.reference_band = self._normalize_reference_band(reference_band)
        self.time_tolerance = self._normalize_time_tolerance(time_tolerance)
        self.range_tolerance = float(range_tolerance)

        radar_a = self._coerce_radar(first)
        radar_b = self._coerce_radar(second)
        self.ka, self.w = self._assign_band_roles(radar_a, radar_b)
        self._dataset: xr.Dataset | None = None

    @staticmethod
    def _normalize_reference_band(reference_band: str) -> str:
        """Normalize and validate the requested reference band."""
        band = reference_band.strip().upper()
        if band not in {"KA", "W"}:
            raise ValueError("reference_band must be either 'Ka' or 'W'.")
        return "Ka" if band == "KA" else "W"

    @staticmethod
    def _normalize_time_tolerance(
        value: np.timedelta64 | timedelta | str,
    ) -> np.timedelta64:
        """Normalize a time-alignment tolerance to ``numpy.timedelta64``."""
        if isinstance(value, np.timedelta64):
            return value
        if isinstance(value, timedelta):
            return np.timedelta64(int(value.total_seconds() * 1_000_000_000), "ns")
        if isinstance(value, str):
            return np.timedelta64(value)
        raise TypeError(
            "time_tolerance must be a numpy.timedelta64, datetime.timedelta, or str."
        )

    @staticmethod
    def _coerce_radar(value: rpg | str | Path) -> rpg:
        """Return an :class:`rpg` instance from a path or existing object."""
        if isinstance(value, rpg):
            return value
        return rpg(value)

    @staticmethod
    def _assign_band_roles(radar_a: rpg, radar_b: rpg) -> tuple[rpg, rpg]:
        """Assign Ka and W roles to two single-band radar objects.

        Raises
        ------
        ValueError
            If the inputs do not contain exactly one Ka-band radar and one
            W-band radar.
        """
        bands = {radar_a.band, radar_b.band}
        if bands != {"Ka", "W"}:
            raise ValueError(
                "dual_rpg requires one Ka-band radar and one W-band radar input."
            )
        ka_radar = radar_a if radar_a.band == "Ka" else radar_b
        w_radar = radar_a if radar_a.band == "W" else radar_b
        return ka_radar, w_radar

    @property
    def paths(self) -> dict[str, Path]:
        """Return the source paths for the Ka and W datasets."""
        return {"Ka": self.ka.path, "W": self.w.path}

    @property
    def level(self) -> int:
        """Return the common processing level of the paired radars.

        Raises
        ------
        ValueError
            If the two inputs do not share the same level.
        """
        levels = {self.ka.level, self.w.level}
        if len(levels) != 1:
            raise ValueError("Ka and W inputs must have the same processing level.")
        return self.ka.level

    @property
    def type(self) -> str:
        """Return the common scan type of the paired radars.

        Raises
        ------
        ValueError
            If the two inputs do not share the same scan type.
        """
        types = {self.ka.type, self.w.type}
        if len(types) != 1:
            raise ValueError("Ka and W inputs must have the same scan type.")
        return self.ka.type

    @property
    def reference_radar(self) -> rpg:
        """Return the single-band radar used as the reference grid."""
        return self.w if self.reference_band == "W" else self.ka

    @property
    def secondary_radar(self) -> rpg:
        """Return the radar aligned onto the reference-grid coordinates."""
        return self.ka if self.reference_band == "W" else self.w

    def _mergeable_variables(self, dataset: xr.Dataset) -> xr.Dataset:
        """Select variables with dimensions supported by the dual merge.

        Parameters
        ----------
        dataset : xarray.Dataset
            Single-band dataset from which mergeable variables are selected.

        Returns
        -------
        xarray.Dataset
            Dataset subset containing only variables whose dimensions are
            compatible with the dual-frequency combined schema.
        """
        selected_variables = [
            name
            for name, data_array in dataset.data_vars.items()
            if frozenset(data_array.dims) in self._mergeable_dims
        ]
        if not selected_variables:
            raise ValueError("No mergeable variables available for dual-frequency merge.")
        return dataset[selected_variables]

    def _align_to_reference(
        self,
        source: xr.Dataset,
        target_time: xr.DataArray,
        target_range: xr.DataArray,
    ) -> xr.Dataset:
        """Align a source dataset to reference time and range coordinates."""
        aligned = source.reindex(
            time=target_time.values,
            method="nearest",
            tolerance=self.time_tolerance,
        )
        aligned = aligned.reindex(
            range=target_range.values,
            method="nearest",
            tolerance=self.range_tolerance,
        )
        return aligned.assign_coords(time=target_time.values, range=target_range.values)

    @staticmethod
    def _prefix_variables(dataset: xr.Dataset, prefix: str) -> xr.Dataset:
        """Prefix all data variables in a dataset."""
        rename_map = {name: f"{prefix}_{name}" for name in dataset.data_vars}
        return dataset.rename_vars(rename_map)

    @staticmethod
    def _can_build_spectral_dfr(combined: xr.Dataset) -> bool:
        """Check whether a direct spectral DFR field can be stored.

        Returns
        -------
        bool
            ``True`` when Ka and W spectral variables exist and their Doppler
            velocity grids match exactly.
        """
        required_variables = {
            "ka_doppler_spectrum",
            "w_doppler_spectrum",
            "ka_velocity_vectors",
            "w_velocity_vectors",
        }
        if not required_variables.issubset(combined.data_vars):
            return False
        ka_velocity = combined["ka_velocity_vectors"].values
        w_velocity = combined["w_velocity_vectors"].values
        return ka_velocity.shape == w_velocity.shape and np.allclose(
            ka_velocity,
            w_velocity,
            equal_nan=True,
        )

    def _build_dataset(self) -> xr.Dataset:
        """Build the cached dual-frequency dataset.

        Returns
        -------
        xarray.Dataset
            Combined dataset with prefixed Ka/W variables on the reference
            time/range grid plus dual-frequency products such as ``DFR_Ka_W``.
        """
        reference_data = self._mergeable_variables(self.reference_radar.dataset)
        secondary_data = self._mergeable_variables(self.secondary_radar.dataset)
        secondary_on_reference = self._align_to_reference(
            secondary_data,
            reference_data["time"],
            reference_data["range"],
        )

        ka_data = reference_data if self.reference_band == "Ka" else secondary_on_reference
        w_data = reference_data if self.reference_band == "W" else secondary_on_reference

        combined = xr.merge(
            [
                self._prefix_variables(ka_data, "ka"),
                self._prefix_variables(w_data, "w"),
            ],
            compat="override",
            join="exact",
        )
        combined.attrs = {
            "long_name": "Aligned Ka/W dual-frequency RPG dataset",
            "reference_band": self.reference_band,
            "ka_path": self.ka.path.as_posix(),
            "w_path": self.w.path.as_posix(),
            "time_tolerance": str(self.time_tolerance),
            "range_tolerance_m": self.range_tolerance,
        }

        if "ka_dBZe" in combined and "w_dBZe" in combined:
            combined["DFR_Ka_W"] = combined["ka_dBZe"] - combined["w_dBZe"]
            combined["DFR_Ka_W"].attrs = {
                "long_name": "Dual-frequency ratio Ka minus W",
                "units": "dB",
            }

        if "ka_doppler_spectrum" in combined:
            combined["ka_doppler_spectrum_dB"] = retrieve_dBZe(
                combined["ka_doppler_spectrum"],
                "Ka",
            )
        if "w_doppler_spectrum" in combined:
            combined["w_doppler_spectrum_dB"] = retrieve_dBZe(
                combined["w_doppler_spectrum"],
                "W",
            )
        if self._can_build_spectral_dfr(combined):
            combined["spectral_DFR_Ka_W"] = (
                combined["ka_doppler_spectrum_dB"] - combined["w_doppler_spectrum_dB"]
            )
            combined["spectral_DFR_Ka_W"].attrs = {
                "long_name": "Spectral dual-frequency ratio Ka minus W",
                "units": "dB",
            }

        return combined

    @property
    def dataset(self) -> xr.Dataset:
        """Return the cached dual-frequency dataset."""
        if self._dataset is None:
            self._dataset = self._build_dataset()
        return self._dataset

    def to_dataset(self) -> xr.Dataset:
        """Return :attr:`dataset` as an explicit method call."""
        return self.dataset

    def _reference_prefix(self) -> str:
        """Return the variable prefix of the reference band."""
        return "w" if self.reference_band == "W" else "ka"

    def _secondary_prefix(self) -> str:
        """Return the variable prefix of the secondary band."""
        return "ka" if self.reference_band == "W" else "w"

    @staticmethod
    def _interpolate_to_velocity_grid(
        values: np.ndarray,
        source_velocity: np.ndarray,
        target_velocity: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a spectral line onto a target Doppler velocity grid.

        Parameters
        ----------
        values : numpy.ndarray
            Spectral values defined on ``source_velocity``.
        source_velocity : numpy.ndarray
            Original Doppler velocity grid.
        target_velocity : numpy.ndarray
            Target Doppler velocity grid.

        Returns
        -------
        numpy.ndarray
            Values interpolated onto ``target_velocity`` with ``NaN`` outside
            the valid interpolation range.
        """
        valid = np.isfinite(values) & np.isfinite(source_velocity)
        if np.count_nonzero(valid) < 2:
            return np.full(target_velocity.shape, np.nan, dtype=float)

        source_velocity = source_velocity[valid]
        values = values[valid]
        order = np.argsort(source_velocity)
        source_velocity = source_velocity[order]
        values = values[order]
        return np.interp(
            target_velocity,
            source_velocity,
            values,
            left=np.nan,
            right=np.nan,
        )

    def _spectral_dfr_line(
        self,
        target_time: datetime | np.datetime64,
        target_range: float,
    ) -> xr.DataArray:
        """Compute a spectral DFR line at one time/range point.

        Parameters
        ----------
        target_time : datetime.datetime or numpy.datetime64
            Time to select. The nearest available sample is used.
        target_range : float
            Range gate in meters. The nearest available gate is used.

        Returns
        -------
        xarray.DataArray
            One-dimensional spectral DFR line in dB on the reference-band
            Doppler velocity grid.

        Notes
        -----
        When the Ka and W Doppler velocity grids differ, the secondary-band
        spectrum is interpolated onto the reference-band grid before the
        ``Ka - W`` difference is computed.
        """
        reference_prefix = self._reference_prefix()
        secondary_prefix = self._secondary_prefix()
        dataset = self.dataset
        reference_chirp_number = int(
            dataset[f"{reference_prefix}_chirp_number"]
            .sel(range=target_range, method="nearest")
            .item()
        )
        secondary_chirp_number = int(
            dataset[f"{secondary_prefix}_chirp_number"]
            .sel(range=target_range, method="nearest")
            .item()
        )
        reference_velocity = (
            dataset[f"{reference_prefix}_velocity_vectors"]
            .sel(chirp=reference_chirp_number)
            .values
        )
        secondary_velocity = (
            dataset[f"{secondary_prefix}_velocity_vectors"]
            .sel(chirp=secondary_chirp_number)
            .values
        )
        reference_line = dataset[f"{reference_prefix}_doppler_spectrum_dB"].sel(
            time=target_time,
            range=target_range,
            method="nearest",
        )
        secondary_line = dataset[f"{secondary_prefix}_doppler_spectrum_dB"].sel(
            time=target_time,
            range=target_range,
            method="nearest",
        )
        secondary_on_reference = self._interpolate_to_velocity_grid(
            secondary_line.values,
            secondary_velocity,
            reference_velocity,
        )
        if reference_prefix == "ka":
            dfr_values = reference_line.values - secondary_on_reference
        else:
            dfr_values = secondary_on_reference - reference_line.values

        line = xr.DataArray(
            dfr_values,
            coords={"spectrum": reference_velocity},
            dims=("spectrum",),
            attrs={
                "long_name": "Spectral dual-frequency ratio Ka minus W",
                "units": "dB",
            },
        )
        return line.where(np.isfinite(line), np.nan)

    @staticmethod
    def _maybe_add_legend(ax: Axes, **kwargs) -> None:
        """Add a legend only when visible labeled artists are present."""
        handles, labels = ax.get_legend_handles_labels()
        visible_labels = [
            label for label in labels if label and not label.startswith("_")
        ]
        if visible_labels:
            ax.legend(
                ncol=kwargs.get("ncol", 2),
                loc="upper right",
                fontsize=kwargs.get("fontsize_legend", kwargs.get("fontsize", 8)),
            )

    def plot_spectral_dfr_by_range(
        self,
        target_time: datetime | np.datetime64,
        ranges: list[float] | np.ndarray,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Plot spectral DFR lines for multiple heights at one time.

        Parameters
        ----------
        target_time : datetime.datetime or numpy.datetime64
            Time to select. The nearest available sample is used.
        ranges : list[float] or numpy.ndarray
            Heights in meters to plot. Nearest available range gates are used.
        **kwargs
            Plot options such as ``figsize``, ``color_list``, ``title``,
            ``output_dir``, ``savefig``, and ``dpi``.

        Returns
        -------
        tuple[matplotlib.figure.Figure, pathlib.Path or None]
            Figure handle and saved filepath when applicable.

        Raises
        ------
        ValueError
            If no ranges are provided or if saving is requested without
            ``output_dir``.
        """
        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)

        range_values = np.unique(
            [
                float(self.dataset.sel(range=range_value, method="nearest").range.item())
                for range_value in ranges
            ]
        )
        if range_values.size == 0:
            raise ValueError("At least one target range must be provided.")

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))
        colors = kwargs.get("color_list", color_list(len(range_values)))

        for idx, range_value in enumerate(range_values):
            line = self._spectral_dfr_line(target_time, float(range_value))
            if np.isfinite(line.values).any():
                line.plot(  # type: ignore[arg-type]
                    ax=ax,
                    color=colors[idx],
                    label=f"{range_value:.0f} m",
                )

        ax.set_xlabel("Doppler velocity, [m/s]")
        ax.set_ylabel("Spectral DFR Ka-W, [dB]")
        ax.set_title(
            kwargs.get(
                "title",
                f"Spectral DFR Ka-W at {target_time:%Y-%m-%d %H:%M:%S}",
            )
        )
        ax.axvline(0.0, color="black", linestyle="--")
        self._maybe_add_legend(ax, **kwargs)
        fig.tight_layout()

        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir")
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            output_dir = Path(output_dir)
            filepath = output_dir / (
                f"dual_rpg_spectral_dfr_{target_time:%Y%m%dT%H%M%S}_"
                f"{range_values[0]:.0f}_{range_values[-1]:.0f}.png"
            )
            fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
        return fig, filepath

    def plot_dfr_quicklook(
        self,
        time_limits: tuple[datetime | np.datetime64, datetime | np.datetime64] | None = None,
        range_limits: tuple[float, float] | None = None,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Plot a time-height quicklook of ``DFR_Ka_W``.

        Parameters
        ----------
        time_limits : tuple[datetime.datetime or numpy.datetime64, datetime.datetime or numpy.datetime64], optional
            Time interval to display. When omitted, all available times are
            shown.
        range_limits : tuple[float, float], optional
            Range interval in meters. When omitted, all available ranges are
            shown.
        **kwargs
            Plot options such as ``figsize``, ``cmap``, ``vmin``, ``vmax``,
            ``title``, ``output_dir``, ``savefig``, and ``dpi``.

        Returns
        -------
        tuple[matplotlib.figure.Figure, pathlib.Path or None]
            Figure handle and saved filepath when applicable.

        Raises
        ------
        ValueError
            If ``DFR_Ka_W`` is unavailable, the requested limits select no
            data, or saving is requested without ``output_dir``.
        """
        if "DFR_Ka_W" not in self.dataset:
            raise ValueError("DFR_Ka_W is not available in the dual-frequency dataset.")

        data = self.dataset["DFR_Ka_W"]
        if time_limits is not None:
            start_time, end_time = time_limits
            if isinstance(start_time, np.datetime64):
                start_time = parse_datetime(start_time)
            if isinstance(end_time, np.datetime64):
                end_time = parse_datetime(end_time)
            data = data.sel(time=slice(start_time, end_time))
        if range_limits is not None:
            data = data.sel(range=slice(*range_limits))
        if data.size == 0 or data.sizes.get("time", 0) == 0 or data.sizes.get("range", 0) == 0:
            raise ValueError("No DFR_Ka_W data found for the requested quicklook limits.")

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 5)))
        plot_data = data.assign_coords(range=data["range"] / 1e3)
        plot_data.plot(  # type: ignore[arg-type]
            x="time",
            y="range",
            ax=ax,
            cmap=kwargs.get("cmap", "viridis"),
            vmin=kwargs.get("vmin"),
            vmax=kwargs.get("vmax"),
        )
        ax.set_ylabel("Range, [km]")
        ax.set_title(kwargs.get("title", "DFR Ka-W quicklook"))
        fig.tight_layout()

        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir")
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            output_dir = Path(output_dir)
            filepath = output_dir / "dual_rpg_dfr_quicklook.png"
            fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
        return fig, filepath

    def select(
        self,
        target_time: np.datetime64,
        target_range: float,
        *,
        method: str = "nearest",
    ) -> xr.Dataset:
        """Select one point from the combined dataset.

        Parameters
        ----------
        target_time : numpy.datetime64
            Target time to select.
        target_range : float
            Target range in meters to select.
        method : str, default: "nearest"
            Selection method passed to :meth:`xarray.Dataset.sel`.

        Returns
        -------
        xarray.Dataset
            Dataset reduced to the requested time/range point.
        """
        return self.dataset.sel(time=target_time, range=target_range, method=method)
