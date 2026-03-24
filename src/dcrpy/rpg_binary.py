from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import xarray as xr
from rpgpy import read_rpg
from rpgpy.spcutil import scale_spectra

from dcrpy.utils import parse_datetime
from dcrpy.retrieve.retrieve import retrieve_dBZe
from dcrpy.plotting.utils import color_list

class rpg:
    def __init__(self, path: Path):
        self.path = path
        self.type = path.name.split(".")[0].split("_")[-1]
        self.level = int(path.name.split(".")[-1][-1])
        self._raw = None
        self._header = None
        self._band = None
        self._dataset = None

    @property
    def raw(self) -> dict:
        if self._raw is None:
            _, self._raw = read_rpg(self.path)
        return self._raw

    @property
    def header(self) -> dict:
        if self._header is None:
            self._header, _ = read_rpg(self.path)
        return self._header

    @property
    def band(self) -> str:
        if self._band is None:
            if self.header["Freq"] > 75:
                self._band = "W"
            else:
                self._band = "Ka"
        return self._band

    @property
    def dataset(
        self, time_parser: str = "seconds since 2001-01-01 00:00:00 UTC"
    ) -> xr.Dataset:
        if self._dataset == None:
            coords = {
                "time": self.raw["Time"],
                "range": self.header["RAlts"],
                "spectrum": np.arange(self.header["velocity_vectors"].shape[1]),
                "chirp": np.arange(self.header["velocity_vectors"].shape[0]),
            }
            dataset = xr.Dataset(coords=coords)
            dataset["time"].attrs = {"long_name": "time", "units": time_parser}
            dataset = xr.decode_cf(dataset)

            dataset["chirp_number"] = xr.DataArray(
                np.zeros(self.header["RAlts"].size),
                coords={"range": self.header["RAlts"]},
            )
            for idx, start_chirp_ in enumerate(self.header["RngOffs"]):
                dataset["chirp_number"].loc[
                    {"range": dataset.range[start_chirp_:]}
                ] = idx
            dataset = dataset.assign_coords(
                chirp=np.sort(np.unique(dataset["chirp_number"].values.astype(int)))
            )

            dataset['nyquist_velocity'] = (
                ("chirp"),
                self.header["MaxVel"],
            )
            dataset['maximum_range'] = (
                ("chirp"),
                self.header["RangeMax"],
            )
            dataset['minimum_range'] = (
                ("chirp"),
                self.header["RangeMax"],
            )
            dataset['doppler_spectrum_length'] = (
                ("chirp"),
                self.header['SpecN'],
            )
            
            dataset["velocity_vectors"] = (
                ("chirp", "spectrum"),
                self.header["velocity_vectors"],
            )
            dataset["doppler_spectrum_h"] = (
                ("time", "range", "spectrum"),
                self.raw["HSpec"],
            )
            dataset["covariance_spectrum_re"] = (
                ("time", "range", "spectrum"),
                self.raw["ReVHSpec"],
            )
            dataset["doppler_spectrum"] = (
                ("time", "range", "spectrum"),
                scale_spectra(self.raw["TotSpec"], self.header["SWVersion"]),
            )
            dataset["doppler_spectrum_v"] = xr.DataArray(
            dims=dataset["doppler_spectrum"].dims,
            data=dataset["doppler_spectrum"].values
            - dataset["doppler_spectrum_h"].values
            - 2 * dataset["covariance_spectrum_re"].values,
            )
            small_value = 1e-10
            dataset["sZDR"] = xr.DataArray(
                dims=dataset["doppler_spectrum_h"].dims,
                data=10 * np.log10(np.where(dataset["doppler_spectrum_h"].values > 0, dataset["doppler_spectrum_h"].values, small_value))
        - 10 * np.log10(np.where(dataset["doppler_spectrum_v"].values > 0, dataset["doppler_spectrum_v"].values, small_value))
)
            dataset["sZDRmax"] = dataset["sZDR"].max(dim="spectrum")
            self._dataset = dataset
        return self._dataset
    

    def plot_spectrum(
        self, target_time: datetime | np.datetime64, target_range: float, **kwargs
    ) -> tuple[Figure, Path | None]:
        """Generates a plot of the doppler spectrum at a specific time and range.

        Args:

            - target_time (datetime | np.datetime64): The time for the plot.
            - target_range (float):  The range for the plot.

        Returns:

            - tuple[Figure, Path | None]: The figure handle and the path to the saved file (None if `savefig = False`).
        """

        fig = kwargs.get("fig", None) 
        if fig is None:
            fig, ax = plt.subplots(figsize=(10, 7))
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
            label = f"{time_str}"

        data = self.dataset
        chirp_number = int(
            data["chirp_number"].sel(range=target_range, method="nearest").values.item()
        )
        sdata = data.sel(time=target_time, range=target_range, method="nearest")
        velocity_vectors = sdata["velocity_vectors"].sel(chirp=chirp_number).values
        sdata = sdata.assign_coords(spectrum=velocity_vectors)

        # Convert to dBZe
        sdata["doppler_spectrum_dBZe"] = retrieve_dBZe(sdata["doppler_spectrum"], self.band)
        sdata["doppler_spectrum_dBZe"].attrs = {
            "long_name": "Power density",
            "units": "dB",
        }
        if not np.isnan(sdata["doppler_spectrum_dBZe"].values).all():
            sdata["doppler_spectrum_dBZe"].plot(
                ax=ax, color=kwargs.get("color", "black"), label=label
            ) #type: ignore

        if kwargs.get("velocity_limits", None) is not None:
            ax.set_xlim(*kwargs.get("velocity_limits"))
        else:
            nyquist_velocity = data["nyquist_velocity"].sel(chirp=chirp_number).values.item()
            ax.set_xlim(-nyquist_velocity, nyquist_velocity)

        ax.set_xlabel("Doppler velocity, [m/s]")
        ax.set_ylabel("Power density, [dB]")
        ax.set_title(f"Time: {str(target_time).split('.')[0]}, Range: {target_range}")

        # Add vertical lines at 0
        ax.axvline(x=0, color="black", linestyle="--")
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)

        filepath = None
        if fig is not None:
            fig.tight_layout()
            output_dir = kwargs.get("output_dir", None)
            if output_dir is not None:
                filepath = output_dir / f"{self.path.stem}_spectrum_{target_time:%Y%m%dT%H%M}_{target_range:.0f}.png"                
                fig.savefig(filepath, dpi=300)
        return fig, filepath

    def plot_spectra_by_range(
        self,
        target_time: datetime | np.datetime64,
        range_slice: tuple[float, float] | list[float],
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Generates a plot of the doppler spectra at a specific time and time slice.

        Args:

            - target_time (float): time to plot the spectra.
            - range_slice (tuple[float, float]): time slice to plot the spectra
            - kwargs: additional arguments to pass to the plot_spectrum method (e.g., savefig | output_dir).

        Returns:
            - tuple[Figure, Path | None]: The figure handle and the path to the saved file (None if `savefig = False`).
        
        """
        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)
        
        original_range_slice = range_slice
        
        data = self.dataset.copy()
        
        if isinstance(range_slice, list):
            _range_list = range_slice
            range_list = np.unique(
                [
                    data.sel(range=range_, method="nearest").range.values
                    for range_ in _range_list
                ]
            )
            range_slice = (range_list[0], range_list[-1])
        else:
            range_slice = range_slice
            range_list = data.range.sel(range=slice(*range_slice)).values
        
        data = data.sel(time=target_time, method="nearest")
        data = data.sel(range=slice(*range_slice))
        
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))        
        colors = kwargs.get('color_list', color_list(len(range_list)))
        if len(data.range) == 0:
            raise ValueError("No data found for the given range slice.")
        for idx, range_ in enumerate(range_list):
            self.plot_spectrum(
                target_time,
                range_,
                **{"color": colors[idx]},
                **{
                    "fig": fig,
                    "savefig": False,
                    "velocity_limits": kwargs.get("velocity_limits", None),
                    "label_type": "range",
                },
            )
        default_title = f"Time: {target_time:%Y%m%dT%H:%M:%S} | Range: [{range_slice[0]/1e3:.2f} - {range_slice[1]/1e3:.2f}] km"
        ax.tick_params(axis='both', which='major', labelsize=kwargs.get("fontsize_axis", 12))
        ax.set_title(kwargs.get("title", default_title), fontdict={"fontsize": kwargs.get("fontsize_title", 12)})
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=kwargs.get("fontsize_legend", 12))
        ax.set_xlabel("Doppler velocity, [m/s]", fontsize=kwargs.get("fontsize_labels", 12))
        ax.set_ylabel("Power density, [dB]", fontsize=kwargs.get("fontsize_labels", 12))

        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir", None)
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            breakpoint()
            filepath = output_dir / f"{self.path.stem}_spectra_{target_time:%Y%m%dT%H%M%S}_{original_range_slice[0]:.0f}_{original_range_slice[-1]:.0f}.png"
            fig.savefig(filepath, dpi=300)
        return fig, filepath
    
    
    def plot_spectra_by_time(
        self,
        target_range: float,
        time_slice: tuple[datetime, datetime] | list[datetime],
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Generates a plot of the doppler spectra at a specific range and time slice.

        Args:

            - target_range (float): range to plot the spectra.
            - time_slice (tuple[datetime, datetime]): time slice to plot the spectra
            - kwargs: additional arguments to pass to the plot_spectrum method (e.g., savefig | output_dir).

        Returns:
            - tuple[Figure, Path | None]: The figure handle and the path to the saved file (None if `savefig = False`).
        """
        original_time_slice = time_slice   
        if all(isinstance(t, np.datetime64) for t in original_time_slice):
            original_time_slice = (parse_datetime(original_time_slice[0]), parse_datetime(original_time_slice[1]))
        
        data = self.dataset.copy()
        
        if isinstance(time_slice, list):
            _time_list = time_slice
            time_list = np.unique(
                [
                    data.sel(time=time_, method="nearest").time.values
                    for time_ in _time_list
                ]
            )
            time_slice = (time_list[0], time_list[-1])
            # Find the time_list in the data.time and create a new time_list with the values found using .sel(method="nearest")
        else:
            time_slice = time_slice
            time_list = data.time.sel(time=slice(*time_slice)).values
        
        data = data.sel(range=target_range, method="nearest")
        data = data.sel(time=slice(*time_slice))
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 7)))        
        colors = kwargs.get('color_list', color_list(len(time_list)))
        if len(data.time) == 0:
            raise ValueError("No data found for the given time slice.")
        for idx, time_ in enumerate(time_list):
            self.plot_spectrum(
                time_,
                target_range,
                **{"color": colors[idx]},
                **{
                    "fig": fig,
                    "savefig": False,
                    "velocity_limits": kwargs.get("velocity_limits", None),
                    "label_type": "time",
                },
            )
        
        ax.set_title(
            f"Range: {target_range} | Period: {original_time_slice[0]:%Y%m%d} {original_time_slice[0]:%H:%M:%S} - {original_time_slice[-1]:%H:%M:%S}"
        )
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)

        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir", None)
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            filepath = output_dir / f"{self.path.stem}_spectra_{original_time_slice[0]:%Y%m%dT%H%M}_{original_time_slice[-1]:%Y%m%dT%H%M}_{target_range}.png"
            fig.savefig(filepath, dpi=300)
        return fig, filepath
    

    def plot_2D_spectrum(
        self,
        target_time: np.datetime64 | datetime,
        range_limits: tuple[float, float] | None = None,
        vmin: float = 0,
        vmax: float = 1,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """Generates a 2D plot of the doppler spectrum at a specific time.

        Args:
            target_time (np.datetime64 | datetime): The time for the plot.
            range_limits (tuple[float, float], optional): Range limits for the plot. Defaults (None) to the minimum and maximum range in the dataset.
            vmin (float, optional): _description_. Defaults to 0.
            vmax (float, optional): _description_. Defaults to 1.

        Returns:
            tuple[Figure, Path | None]: _description_
        """

        """
        Generates a range-velocity plot based on the dataset and a specific time.

        Args:

            - dataset (pandas.DataFrame): The dataset containing the required data.
            - time_str (str): The time string for the desired plot time (e.g., '20210913T112500.0').
            - min_value (float, optional): The minimum value to consider for the doppler_spectrum_dBZe plot. Default is 0.01.
            - power_spectrum_limits (tuple, optional): The limits for the power spectrum plot. Default sets limits to autoscale.

        Returns:

            - None
        """
        if isinstance(target_time, np.datetime64):
            target_time = parse_datetime(target_time)
        
        data = self.dataset.copy()
        if range_limits is None:
            range_limits = (data.range.min().values, data.range.max().values)
        range_limits = (range_limits[0] / 1e3, range_limits[1] / 1e3)
        
        data["range"] = data["range"] / 1e3
        data = data.sel(time=target_time, method="nearest")
        data = data.sel(range=slice(*range_limits))
        chirps = np.sort(np.unique(data["chirp_number"].values).astype(int))
        number_of_chirps = len(chirps)
        data = data.sel(chirp=chirps)

        data["doppler_spectrum_dBZe"] = retrieve_dBZe(data["doppler_spectrum"], self.band)
        data["doppler_spectrum_dBZe"].attrs = {
            "long_   ame": "Power density",
            "units": "dB",
        }

        chirp_info = {"range_limits": {}, "height_ratio": {}}

        for chirp_ in chirps:
            min_range = data.range[data["chirp_number"].values == chirp_].min().item()
            max_range = data.range[data["chirp_number"].values == chirp_].max().item()
            chirp_info["range_limits"][chirp_] = (min_range, max_range)
            chirp_info["height_ratio"][chirp_] = (
                100 * (max_range - min_range) / (range_limits[1] - range_limits[0])
            )

        height_ratios = np.flip(
            [chirp_info["height_ratio"][chirp_] for chirp_ in chirps]
        )

        fig, axes = plt.subplots(
            number_of_chirps,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": height_ratios},
        )  # Adjust height_ratios if chirps change
        if isinstance(axes, Axes):
            axes = np.array([axes])

        axes = np.flip(
            axes
        )  # Reverse the order of the axes to plot the highest chirp first

        cm = []
        for idx, chirp_ in enumerate(data.chirp.values):
            x_vals = data["velocity_vectors"].sel(chirp=chirp_).values
            nyquist_velocity = data["nyquist_velocity"].sel(chirp=chirp_).values
            # Plot the doppler_spectrum
            cm_ = axes[idx].imshow(
                data["doppler_spectrum_dBZe"],
                aspect="auto",
                extent=[x_vals[0], x_vals[-1], range_limits[-1], range_limits[0]],
                cmap=kwargs.get("cmap", "jet"),
            )
            axes[idx].set_ylim(*chirp_info["range_limits"][chirp_])
            axes[idx].axvline(x=nyquist_velocity, color="gray", linestyle="--")
            axes[idx].axvline(x=-nyquist_velocity, color="gray", linestyle="--")
            cm.append(cm_)

            vmin, vmax = kwargs.get("power_spectrum_limits", ( data["nyquist_velocity"].min().item(), data["nyquist_velocity"].max().item()))
            for cm_ in cm:
                cm_.set_clim(vmin, vmax)

        axes[0].set_xlabel("Doppler Velocity, [m/s]")
        nyquist_velocity_limits = (
            -data["nyquist_velocity"].max().values.item(),
            data["nyquist_velocity"].max().values.item(),
        )
        for ax_ in axes:
            ax_.set_facecolor("white")
            ax_.set_xlim(*nyquist_velocity_limits)
            ax_.set_ylabel("Height, [m]")
            ax_.axvline(x=0, color="black", linestyle="--")
            ax_.minorticks_on()
            ax_.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
            ax_.grid(
                which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5
            )

        for ax_ in axes[1:]:
            ax_.set_xticklabels([])

        fig.suptitle(
            f'2D Doppler spectrum at {str(target_time).split(".")[0]}',
            fontsize=16,
        )

        plt.subplots_adjust(hspace=0.05)
        cax = fig.add_axes(
            (0.85, 0.15, 0.04, 0.7)
        )  # Adjust the width value to make the colorbar wider
        if kwargs.get("colorbar_label", None) is not None:
            fig.colorbar(cm[0], cax=cax, label=kwargs.get("colorbar_label"))
        else:
            if isinstance(cm, list):
                fig.colorbar(cm[0], cax=cax, label="linear Ze")
            else:
                fig.colorbar(cm, cax=cax, label="linear Ze")

        fig.subplots_adjust(left=0.1, right=0.82, bottom=0.10, top=0.9, wspace=0.2)

        # Adjust the right value to accommodate the wider colorbar
        # fig.tight_layout()

        output_dir = kwargs.get("output_dir", None)
        filepath = None
        if output_dir is not None:            
            filepath = output_dir / f"{self.path.stem}_2d-spectrum_{target_time:%Y%m%dT%H%M}.png"
            fig.savefig(filepath, dpi=300)
        return fig, filepath


    def plot_profile(
        self,
        target_times: datetime | np.datetime64 | list[datetime] | tuple[datetime, datetime],
        range_limits: tuple[float, float],
        variable: str,
        **kwargs,
    ) -> tuple[Figure, Path | None]:
        """
        Plots a profile of a specified variable over a given time period and range limits.
        Parameters
        ----------
        target_times : datetime | np.datetime64 | list[datetime] | tuple[datetime, datetime]
            The target times for which the profile is to be plotted. It can be a single datetime, 
            a numpy datetime64, a list of datetimes, or a tuple specifying a start and end datetime.
        range_limits : tuple[float, float]
            The range limits (in meters) for the profile plot.
        variable : str
            The variable to be plotted.
        **kwargs : dict, optional
            Additional keyword arguments for customization:
            - fig : Figure, optional
                A pre-existing figure to plot on. If not provided, a new figure is created.
            - figsize : tuple, optional
                Size of the figure (default is (5, 7)).
            - color_list : list, optional
                List of colors to use for the plot lines.
            - range_limits : tuple, optional
                Limits for the y-axis (range) in kilometers.
            - variable_limits : tuple, optional
                Limits for the x-axis (variable values).
            - ncol : int, optional
                Number of columns for the legend (default is 2).
            - savefig : bool, optional
                Whether to save the figure (default is True).
            - output_dir : Path, optional
                Directory to save the figure if savefig is True.
        Returns
        -------
        tuple[Figure, Path | None]
            The figure object and the path to the saved figure file (if savefig is True).
        Raises
        ------
        ValueError
            If target_times is not a datetime, np.datetime64, tuple, or list.
            If output_dir is not provided when savefig is True.
        """
        fig = kwargs.get("fig", None)
        if fig is None:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (5, 7)))
        else:            
            ax = fig.get_axes()[0]  
            fig.savefig('testka.png', dpi=300)

        original_target_times = target_times        

        data = self.dataset.copy()

        if isinstance(target_times, np.datetime64) or isinstance(target_times, datetime):
            time_list = [data.sel(time=target_times, method="nearest").time.values]
            title_str = f"Time: {str(time_list[0]).split('.')[0]}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_profile_{original_target_times:%Y%m%dT%H%M}.png",
            )
        elif isinstance(target_times, tuple):            
            time_list = data.sel(time=slice(*target_times)).time.values

            title_str = f"Period: {target_times[0]:%Y-%m-%d} {target_times[0]:%H:%M:%S} - {target_times[-1]:%H:%M:%S}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_profile_{target_times[0]:%Y%m%dT%H%M}_{target_times[-1]:%Y%m%dT%H%M}.png",
            )
        elif isinstance(target_times, list):
            breakpoint()
            _time_list = target_times
            time_list = np.unique(
                [
                    data.sel(time=time_, method="nearest").time.values
                    for time_ in _time_list
                ]
            )
            title_str = f"Period:  {target_times[0]:%Y-%m-%d} {target_times[0]:%H:%M:%S} - {target_times[-1]:%H:%M:%S}"
            filename = self.path.name.replace(
                ".nc",
                f"_{variable}_profile_{target_times[0]:%Y%m%dT%H%M}_{target_times[-1]:%Y%m%dT%H%M}.png",
            )
        else:
            raise ValueError("target_times must be a datetime, np.datetime64, tuple or list.")
        
        breakpoint()
        
        data = data.sel(range=slice(*range_limits))
        colors = kwargs.get('color_list', color_list(len(time_list)))
        data["range"] = data["range"] / 1e3
        for idx, time_ in enumerate(time_list):
            data[variable].sel(time=time_).plot(y='range', ax=ax, color=colors[idx], label=f"{parse_datetime(time_):%H:%M:%S}") #type: ignore
        if 'range_limits' in kwargs:
            range_limits = kwargs.get('range_limits', [0, 10000.])/1e3
            ax.set_ylim(range_limits)
        if 'variable_limits' in kwargs:            
            ax.set_xlim(kwargs.get('variable_limits'))
        
        ax.set_ylabel("Range, [km]")
        ax.set_xlabel(f"{data[variable].attrs['long_name']}, [{data[variable].attrs['units']}]")
        ax.grid(which="major", color="gray", linestyle="--", linewidth=0.5)
        ax.grid(
            which="minor", axis="x", color="gray", linestyle=":", linewidth=0.5
        )

        ax.set_title(title_str)
        ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)        
        fig.tight_layout()
        filepath = None
        if kwargs.get("savefig", True):
            output_dir = kwargs.get("output_dir", None)
            if output_dir is None:
                raise ValueError("output_dir must be provided if savefig is True.")
            filepath = output_dir / filename
            fig.savefig(filepath, dpi=300)
        return fig, filepath
