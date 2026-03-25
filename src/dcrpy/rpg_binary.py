from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import xarray as xr
from rpgpy import read_rpg
from rpgpy.spcutil import scale_spectra

from dcrpy.plotting.utils import color_list
from dcrpy.retrieve.moments import ze_from_spectrum
from dcrpy.retrieve.retrieve import (
    retrieve_dBZe,
    retrieve_doppler_spectrum_v,
    retrieve_spectral_ZDR,
)
from dcrpy.utils import parse_datetime


class rpg:
    _spectral_power_variables = frozenset(
        {"doppler_spectrum", "doppler_spectrum_h", "doppler_spectrum_v"}
    )
    _spectral_variable_labels = {
        "doppler_spectrum": "total",
        "doppler_spectrum_h": "horizontal",
        "doppler_spectrum_v": "vertical",
    }

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.type = self.path.name.split(".")[0].split("_")[-1]
        self.level = int(self.path.name.split(".")[-1][-1])
        self._raw: dict | None = None
        self._header: dict | None = None
        self._band: str | None = None
        self._dataset: xr.Dataset | None = None

    def _ensure_loaded(self) -> None:
        if self._header is None or self._raw is None:
            self._header, self._raw = read_rpg(self.path)

    @property
    def raw(self) -> dict:
        self._ensure_loaded()
        return self._raw  # type: ignore[return-value]

    @property
    def header(self) -> dict:
        self._ensure_loaded()
        return self._header  # type: ignore[return-value]

    @property
    def band(self) -> str:
        if self._band is None:
            frequency = float(np.asarray(self.header["Freq"]).item())
            self._band = "W" if frequency > 75 else "Ka"
        return self._band

    def _validate_source_shapes(self) -> None:
        required_header_keys = (
            "Freq",
            "MaxVel",
            "RAlts",
            "RangeMax",
            "RngOffs",
            "SWVersion",
            "SpecN",
            "velocity_vectors",
        )
        required_raw_keys = ("HSpec", "ReVHSpec", "Time", "TotSpec")

        for key in required_header_keys:
            if key not in self.header:
                raise KeyError(f"Missing header key: {key}")
        for key in required_raw_keys:
            if key not in self.raw:
                raise KeyError(f"Missing raw key: {key}")

        velocity_vectors = np.asarray(self.header["velocity_vectors"])
        if velocity_vectors.ndim != 2:
            raise ValueError("header['velocity_vectors'] must be a 2D array.")
        n_chirps, n_spectrum = velocity_vectors.shape
        n_time = np.asarray(self.raw["Time"]).shape[0]
        n_range = np.asarray(self.header["RAlts"]).shape[0]

        chirp_metadata = {
            "MaxVel": np.asarray(self.header["MaxVel"]),
            "RangeMax": np.asarray(self.header["RangeMax"]),
            "RngOffs": np.asarray(self.header["RngOffs"], dtype=int),
            "SpecN": np.asarray(self.header["SpecN"]),
        }
        for name, values in chirp_metadata.items():
            if values.shape[0] != n_chirps:
                raise ValueError(
                    f"header['{name}'] length {values.shape[0]} does not match the "
                    f"number of chirps {n_chirps}."
                )

        start_indices = chirp_metadata["RngOffs"]
        if np.any(start_indices < 0) or np.any(start_indices >= n_range):
            raise ValueError("header['RngOffs'] contains out-of-bounds indices.")
        if np.any(np.diff(start_indices) < 0):
            raise ValueError("header['RngOffs'] must be sorted in ascending order.")

        expected_shape = (n_time, n_range, n_spectrum)
        for name in ("HSpec", "ReVHSpec", "TotSpec"):
            actual_shape = np.asarray(self.raw[name]).shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"raw['{name}'] has shape {actual_shape}, expected {expected_shape}."
                )

    def _build_dataset(self) -> xr.Dataset:
        self._validate_source_shapes()

        time_values = np.asarray(self.raw["Time"])
        range_values = np.asarray(self.header["RAlts"])
        velocity_vectors = np.asarray(self.header["velocity_vectors"])
        start_indices = np.asarray(self.header["RngOffs"], dtype=int)
        n_chirps, n_spectrum = velocity_vectors.shape

        coords = {
            "time": time_values,
            "range": range_values,
            "spectrum": np.arange(n_spectrum, dtype=int),
            "chirp": np.arange(n_chirps, dtype=int),
        }
        dataset = xr.Dataset(coords=coords)
        dataset["time"].attrs = {
            "long_name": "time",
            "units": "seconds since 2001-01-01 00:00:00 UTC",
        }
        dataset = xr.decode_cf(dataset)

        chirp_number = np.zeros(range_values.size, dtype=np.int16)
        for idx, start_index in enumerate(start_indices):
            chirp_number[start_index:] = idx
        dataset["chirp_number"] = xr.DataArray(
            chirp_number,
            coords={"range": dataset["range"]},
            dims=("range",),
            attrs={"long_name": "chirp index for each range gate"},
        )

        minimum_range = np.asarray(
            self.header.get("RangeMin", range_values[start_indices]),
            dtype=float,
        )
        maximum_range = np.asarray(self.header["RangeMax"], dtype=float)

        dataset["nyquist_velocity"] = (
            ("chirp",),
            np.asarray(self.header["MaxVel"], dtype=float),
        )
        dataset["maximum_range"] = (("chirp",), maximum_range)
        dataset["minimum_range"] = (("chirp",), minimum_range)
        dataset["doppler_spectrum_length"] = (
            ("chirp",),
            np.asarray(self.header["SpecN"], dtype=int),
        )
        dataset["velocity_vectors"] = (
            ("chirp", "spectrum"),
            velocity_vectors,
        )
        dataset["doppler_spectrum_h"] = (
            ("time", "range", "spectrum"),
            np.asarray(self.raw["HSpec"]),
        )
        dataset["covariance_spectrum_re"] = (
            ("time", "range", "spectrum"),
            np.asarray(self.raw["ReVHSpec"]),
        )
        dataset["doppler_spectrum"] = (
            ("time", "range", "spectrum"),
            scale_spectra(np.asarray(self.raw["TotSpec"]), self.header["SWVersion"]),
        )

        dataset["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(dataset)
        dataset["sZDR"] = retrieve_spectral_ZDR(dataset)
        dataset["sZDRmax"] = dataset["sZDR"].max(dim="spectrum", keep_attrs=True)
        dataset["Ze"] = ze_from_spectrum(dataset, variable="doppler_spectrum")
        dataset["Ze"].attrs = {
            "long_name": f"{self.band}-band equivalent reflectivity",
            "units": "linear Ze",
        }
        dataset["dBZe"] = retrieve_dBZe(dataset["Ze"], self.band)

        return dataset

    @property
    def dataset(self) -> xr.Dataset:
        if self._dataset is None:
            self._dataset = self._build_dataset()
        return self._dataset

    def _spectral_plot_data(
        self, selected: xr.Dataset, chirp_number: int, variable: str
    ) -> xr.DataArray:
        if variable not in selected:
            raise KeyError(f"{variable} not found in dataset.")

        data_array = selected[variable]
        if "spectrum" not in data_array.dims:
            raise ValueError(f"{variable} is not a spectral variable.")

        velocity_vectors = self.dataset["velocity_vectors"].sel(chirp=chirp_number)
        data_array = data_array.assign_coords(spectrum=velocity_vectors)

        if variable in self._spectral_power_variables:
            data_array = retrieve_dBZe(data_array, self.band)
            data_array.attrs = {"long_name": "Power density", "units": "dB"}

        return data_array.where(np.isfinite(data_array), np.nan)

    def _normalize_variables_to_plot(
        self, variable_to_plot: str | list[str] | tuple[str, ...] | None
    ) -> list[str]:
        if variable_to_plot is None:
            return ["doppler_spectrum"]
        if isinstance(variable_to_plot, str):
            return [variable_to_plot]

        variables = list(variable_to_plot)
        if not variables:
            raise ValueError("variable_to_plot cannot be empty.")
        return variables

    def _resolve_plot_colors(
        self, variables_to_plot: list[str], **kwargs
    ) -> list[str] | np.ndarray:
        if len(variables_to_plot) == 1:
            return [kwargs.get("color", "black")]

        if kwargs.get("color_list") is not None:
            return kwargs["color_list"]

        default_colors = ["black", "tab:blue", "tab:red", "tab:green"]
        if len(variables_to_plot) <= len(default_colors):
            return default_colors[: len(variables_to_plot)]
        return color_list(len(variables_to_plot))

    def _build_plot_label(self, base_label: str, variable: str, multi: bool) -> str:
        if not multi:
            return base_label
        variable_label = self._spectral_variable_labels.get(variable, variable)
        return f"{base_label} | {variable_label}"

    def _set_plot_ylabel(self, ax: Axes, variables_to_plot: list[str]) -> None:
        if len(variables_to_plot) == 1:
            variable = variables_to_plot[0]
            if variable in self._spectral_power_variables:
                ax.set_ylabel("Power density, [dB]")
            else:
                selected = self.dataset[variable]
                ax.set_ylabel(
                    f"{selected.attrs.get('long_name', variable)}, "
                    f"[{selected.attrs.get('units', '?')}]"
                )
            return

        if all(variable in self._spectral_power_variables for variable in variables_to_plot):
            ax.set_ylabel("Power density, [dB]")
            return

        ax.set_ylabel("Value")

    def _maybe_add_legend(self, ax: Axes, **kwargs) -> None:
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

    def plot_spectrum(
        self, target_time: datetime | np.datetime64, target_range: float, **kwargs
    ) -> tuple[Figure, Path | None]:
        fig = kwargs.get("fig")
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))
        else:
            ax = fig.get_axes()[0]

        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)

        time_str = f"{target_time:%H%M%S}"
        range_str = f"{target_range:.2f}"
        label_type = kwargs.get("label_type", "both")
        if label_type == "both":
            label = f"{time_str} | {range_str} m"
        elif label_type == "range":
            label = f"{range_str} m"
        else:
            label = time_str

        data = self.dataset
        variables_to_plot = self._normalize_variables_to_plot(
            kwargs.get("variable_to_plot", "doppler_spectrum")
        )
        colors = self._resolve_plot_colors(variables_to_plot, **kwargs)
        chirp_number = int(
            data["chirp_number"].sel(range=target_range, method="nearest").item()
        )
        selected = data.sel(time=target_time, range=target_range, method="nearest")
        for variable, color in zip(variables_to_plot, colors):
            plot_data = self._spectral_plot_data(selected, chirp_number, variable)
            if np.isfinite(plot_data.values).any():
                plot_data.plot(  # type: ignore[arg-type]
                    ax=ax,
                    color=color,
                    label=self._build_plot_label(
                        label, variable, multi=len(variables_to_plot) > 1
                    ),
                )

        velocity_limits = kwargs.get("velocity_limits")
        if velocity_limits is None:
            nyquist_velocity = data["nyquist_velocity"].sel(chirp=chirp_number).item()
            ax.set_xlim(-nyquist_velocity, nyquist_velocity)
        else:
            ax.set_xlim(*velocity_limits)

        ax.set_xlabel("Doppler velocity, [m/s]")
        self._set_plot_ylabel(ax, variables_to_plot)
        ax.set_title(f"Time: {str(target_time).split('.')[0]}, Range: {target_range}")
        ax.axvline(x=0, color="black", linestyle="--")
        self._maybe_add_legend(ax, **kwargs)

        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir")
            if output_dir is not None:
                output_dir = Path(output_dir)
                fig.tight_layout()
                filepath = output_dir / (
                    f"{self.path.stem}_spectrum_{target_time:%Y%m%dT%H%M}_"
                    f"{target_range:.0f}.png"
                )
                fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
        return fig, filepath

    def plot_spectra_by_range(
        self,
        target_time: datetime | np.datetime64,
        range_slice: tuple[float, float] | list[float],
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)

        original_range_slice = range_slice
        data = self.dataset

        if isinstance(range_slice, list):
            range_list = np.unique(
                [
                    data.sel(range=range_value, method="nearest").range.item()
                    for range_value in range_slice
                ]
            )
            range_slice = (float(range_list[0]), float(range_list[-1]))
        else:
            range_list = data.range.sel(range=slice(*range_slice)).values

        selected = data.sel(time=target_time, method="nearest").sel(
            range=slice(*range_slice)
        )
        if selected.sizes.get("range", 0) == 0:
            raise ValueError("No data found for the given range slice.")

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))
        colors = kwargs.get("color_list", color_list(len(range_list)))
        for idx, range_value in enumerate(range_list):
            self.plot_spectrum(
                target_time,
                float(range_value),
                color=colors[idx],
                fig=fig,
                savefig=False,
                variable_to_plot=kwargs.get("variable_to_plot", "doppler_spectrum"),
                velocity_limits=kwargs.get("velocity_limits"),
                label_type="range",
            )

        default_title = (
            f"Time: {target_time:%Y%m%dT%H:%M:%S} | Range: "
            f"[{range_slice[0] / 1e3:.2f} - {range_slice[1] / 1e3:.2f}] km"
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=kwargs.get("fontsize_axis", 12),
        )
        ax.set_title(
            kwargs.get("title", default_title),
            fontdict={"fontsize": kwargs.get("fontsize_title", 12)},
        )
        self._maybe_add_legend(ax, **kwargs)
        ax.set_xlabel(
            "Doppler velocity, [m/s]", fontsize=kwargs.get("fontsize_labels", 12)
        )
        ax.set_ylabel("Power density, [dB]", fontsize=kwargs.get("fontsize_labels", 12))

        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir")
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            output_dir = Path(output_dir)
            filepath = output_dir / (
                f"{self.path.stem}_spectra_{target_time:%Y%m%dT%H%M%S}_"
                f"{original_range_slice[0]:.0f}_{original_range_slice[-1]:.0f}.png"
            )
            fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
        return fig, filepath

    def plot_spectra_by_time(
        self,
        target_range: float,
        time_slice: tuple[datetime, datetime] | list[datetime],
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        original_time_slice = time_slice
        if all(
            isinstance(time_value, np.datetime64) for time_value in original_time_slice
        ):
            original_time_slice = (
                parse_datetime(original_time_slice[0]),
                parse_datetime(original_time_slice[-1]),
            )

        data = self.dataset
        if isinstance(time_slice, list):
            time_list = np.unique(
                [
                    data.sel(time=time_value, method="nearest").time.values
                    for time_value in time_slice
                ]
            )
            time_slice = (time_list[0], time_list[-1])
        else:
            time_list = data.time.sel(time=slice(*time_slice)).values

        selected = data.sel(range=target_range, method="nearest").sel(
            time=slice(*time_slice)
        )
        if selected.sizes.get("time", 0) == 0:
            raise ValueError("No data found for the given time slice.")

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))
        colors = kwargs.get("color_list", color_list(len(time_list)))
        for idx, time_value in enumerate(time_list):
            self.plot_spectrum(
                time_value,
                target_range,
                color=colors[idx],
                fig=fig,
                savefig=False,
                variable_to_plot=kwargs.get("variable_to_plot", "doppler_spectrum"),
                velocity_limits=kwargs.get("velocity_limits"),
                label_type="time",
            )

        ax.set_title(
            f"Range: {target_range} | Period: {original_time_slice[0]:%Y%m%d} "
            f"{original_time_slice[0]:%H:%M:%S} - {original_time_slice[-1]:%H:%M:%S}"
        )
        self._maybe_add_legend(ax, **kwargs)

        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir")
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            output_dir = Path(output_dir)
            filepath = output_dir / (
                f"{self.path.stem}_spectra_{original_time_slice[0]:%Y%m%dT%H%M}_"
                f"{original_time_slice[-1]:%Y%m%dT%H%M}_{target_range}.png"
            )
            fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
        return fig, filepath

    def plot_2D_spectrum(
        self,
        target_time: np.datetime64 | datetime,
        range_limits: tuple[float, float] | None = None,
        vmin: float = 0,
        vmax: float = 1,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)

        data = self.dataset.sel(time=target_time, method="nearest")
        variable_to_plot = kwargs.get("variable_to_plot", "doppler_spectrum")
        if range_limits is None:
            range_limits = (float(data.range.min().item()), float(data.range.max().item()))

        range_limits_km = (range_limits[0] / 1e3, range_limits[1] / 1e3)
        data = data.assign_coords(range=data["range"] / 1e3)
        data = data.sel(range=slice(*range_limits_km))

        chirps = np.sort(np.unique(data["chirp_number"].values.astype(int)))
        number_of_chirps = len(chirps)
        if number_of_chirps == 0:
            raise ValueError("No data found for the requested time and range limits.")

        chirp_info = {"range_limits": {}, "height_ratio": {}}
        for chirp in chirps:
            chirp_ranges = data["range"].where(data["chirp_number"] == chirp, drop=True)
            min_range = float(chirp_ranges.min().item())
            max_range = float(chirp_ranges.max().item())
            chirp_info["range_limits"][chirp] = (min_range, max_range)
            chirp_info["height_ratio"][chirp] = max(max_range - min_range, 1e-6)

        height_ratios = np.flip(
            [chirp_info["height_ratio"][chirp] for chirp in chirps]
        )
        fig, axes = plt.subplots(
            number_of_chirps,
            1,
            figsize=kwargs.get("figsize", (10, 10)),
            gridspec_kw={"height_ratios": height_ratios},
        )
        if isinstance(axes, Axes):
            axes = np.array([axes])
        axes = np.flip(axes)

        color_limits = kwargs.get("power_spectrum_limits", (vmin, vmax))
        cm = []
        for idx, chirp in enumerate(chirps):
            chirp_ranges = data["range"].where(data["chirp_number"] == chirp, drop=True)
            chirp_data = data.sel(range=chirp_ranges)
            plot_data = self._spectral_plot_data(chirp_data, int(chirp), variable_to_plot)
            plot_data = plot_data.transpose("range", "spectrum")
            x_vals = plot_data["spectrum"].values
            min_range, max_range = chirp_info["range_limits"][chirp]
            nyquist_velocity = data["nyquist_velocity"].sel(chirp=chirp).item()

            cm_ = axes[idx].imshow(
                plot_data.values,
                aspect="auto",
                origin="lower",
                extent=[x_vals[0], x_vals[-1], min_range, max_range],
                cmap=kwargs.get("cmap", "jet"),
                vmin=color_limits[0],
                vmax=color_limits[1],
            )
            axes[idx].axvline(x=nyquist_velocity, color="gray", linestyle="--")
            axes[idx].axvline(x=-nyquist_velocity, color="gray", linestyle="--")
            cm.append(cm_)

        nyquist_velocity_limits = (
            -float(data["nyquist_velocity"].max().item()),
            float(data["nyquist_velocity"].max().item()),
        )
        for ax in axes:
            ax.set_facecolor("white")
            ax.set_xlim(*nyquist_velocity_limits)
            ax.set_ylabel("Height, [km]")
            ax.axvline(x=0, color="black", linestyle="--")
            ax.minorticks_on()
            ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
            ax.grid(which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5)

        axes[-1].set_xlabel("Doppler velocity, [m/s]")
        for ax in axes[:-1]:
            ax.set_xticklabels([])

        fig.suptitle(
            f'2D Doppler spectrum at {str(target_time).split(".")[0]}',
            fontsize=16,
        )
        plt.subplots_adjust(hspace=0.05)
        cax = fig.add_axes((0.85, 0.15, 0.04, 0.7))
        colorbar_label = kwargs.get(
            "colorbar_label",
            f"{plot_data.attrs.get('long_name', variable_to_plot)} "
            f"[{plot_data.attrs.get('units', '?')}]",
        )
        fig.colorbar(cm[0], cax=cax, label=colorbar_label)
        fig.subplots_adjust(left=0.1, right=0.82, bottom=0.10, top=0.9, wspace=0.2)

        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir")
            if output_dir is not None:
                output_dir = Path(output_dir)
                filepath = output_dir / (
                    f"{self.path.stem}_2d-spectrum_{target_time:%Y%m%dT%H%M}.png"
                )
                fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
        return fig, filepath

    def plot_profile(
        self,
        target_times: datetime
        | np.datetime64
        | list[datetime]
        | tuple[datetime, datetime],
        range_limits: tuple[float, float],
        variable: str,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        fig = kwargs.get("fig")
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 7)))
        else:
            ax = fig.get_axes()[0]

        data = self.dataset
        if variable not in data:
            raise KeyError(f"{variable} not found in dataset.")

        if isinstance(target_times, (np.datetime64, datetime)):
            single_time = parse_datetime(target_times)
            time_list = [data.sel(time=target_times, method="nearest").time.values]
            title_str = f"Time: {str(time_list[0]).split('.')[0]}"
            filename = (
                f"{self.path.stem}_{variable}_profile_{single_time:%Y%m%dT%H%M}.png"
            )
        elif isinstance(target_times, tuple):
            start_time = parse_datetime(target_times[0])
            end_time = parse_datetime(target_times[-1])
            time_list = data.sel(time=slice(*target_times)).time.values
            title_str = (
                f"Period: {start_time:%Y-%m-%d} {start_time:%H:%M:%S} - "
                f"{end_time:%H:%M:%S}"
            )
            filename = (
                f"{self.path.stem}_{variable}_profile_{start_time:%Y%m%dT%H%M}_"
                f"{end_time:%Y%m%dT%H%M}.png"
            )
        elif isinstance(target_times, list):
            if len(target_times) == 0:
                raise ValueError("target_times cannot be empty.")
            start_time = parse_datetime(target_times[0])
            end_time = parse_datetime(target_times[-1])
            time_list = np.unique(
                [
                    data.sel(time=time_value, method="nearest").time.values
                    for time_value in target_times
                ]
            )
            title_str = (
                f"Period: {start_time:%Y-%m-%d} {start_time:%H:%M:%S} - "
                f"{end_time:%H:%M:%S}"
            )
            filename = (
                f"{self.path.stem}_{variable}_profile_{start_time:%Y%m%dT%H%M}_"
                f"{end_time:%Y%m%dT%H%M}.png"
            )
        else:
            raise ValueError(
                "target_times must be a datetime, np.datetime64, tuple or list."
            )

        selected = data.sel(range=slice(*range_limits))
        colors = kwargs.get("color_list", color_list(len(time_list)))
        plot_data = selected[variable].assign_coords(range=selected["range"] / 1e3)
        for idx, time_value in enumerate(time_list):
            plot_data.sel(time=time_value).plot(  # type: ignore[arg-type]
                y="range",
                ax=ax,
                color=colors[idx],
                label=f"{parse_datetime(time_value):%H:%M:%S}",
            )

        if "range_limits" in kwargs:
            range_limits_plot = np.asarray(kwargs.get("range_limits"), dtype=float) / 1e3
            ax.set_ylim(range_limits_plot)
        if "variable_limits" in kwargs:
            ax.set_xlim(kwargs.get("variable_limits"))

        ax.set_ylabel("Range, [km]")
        ax.set_xlabel(
            f"{plot_data.attrs.get('long_name', variable)}, "
            f"[{plot_data.attrs.get('units', '?')}]"
        )
        ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
        ax.grid(which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5)
        ax.set_title(title_str)
        self._maybe_add_legend(ax, **kwargs)
        fig.tight_layout()

        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir")
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            output_dir = Path(output_dir)
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
        return fig, filepath
