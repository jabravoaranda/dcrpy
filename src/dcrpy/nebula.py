import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
import xarray as xr
from dcrpy.rpg_nc import rpg
from dcrpy.utils import check_is_netcdf, parse_datetime
from .types import RadarInfoType

def _read_yaml(path: Path):
    with open(path, "r") as stream:
        return yaml.safe_load(stream)

RADAR_INFO: RadarInfoType = _read_yaml(Path(__file__).parent.absolute() / "info.yml")

class nebula():
    def __init__(self, ka_nc_path: Path, ww_nc_path: Path):
        """
        Initializes the Nebula class with the given file paths for ka and ww netCDF files.

        Args:
            ka_nc_path (Path): Path to the ka netCDF file.
            ww_nc_path (Path): Path to the ww netCDF file.
        """
        self.paths = [check_is_netcdf(ka_nc_path), check_is_netcdf(ww_nc_path)]
        self.ka = rpg(ka_nc_path)
        self.ww = rpg(ww_nc_path)
        self._data = None
        self._band = None
    
    @property
    def level(self) -> int:
        if self.ka.level == self.ww.level:
            return self.ka.level
        else:
            raise ValueError("Different type radar measurement cannot be combined.")

    @property
    def type(self) -> str:
        if self.ka.type == self.ww.type:
            return self.ka.type
        else:
            raise ValueError("Different type radar measurement cannot be combined.")

    @property
    def data(self) -> xr.Dataset:
        if self._data is None:
            if self.level == 1:         
                _data = self.ka.data.copy()                   
                _data = _data.drop_vars(_data.variables.keys())

                _data["DWR"] = 10 * np.log10(self.ka.data["Ze"] / self.ww.data["Ze"])
                _data["DWR"].attrs["units"] = "dB"
                _data["DWR"].attrs["long_name"] = "Ka-W DWR"

                _data["DDV"] = self.ka.data["v"] - self.ww.data["v"]
                _data["DDV"].attrs["units"] = "m/s"
                _data["DDV"].attrs["long_name"] = "Ka-W DDV"
            elif self.level == 0:
                pass
                # breakpoint()
                # _data = self.ka.data.copy()
                # _data["DWR"] = 10 * np.log10(self.ka.data["Ze"] / self.ww.data["Ze"])
                # _data["DWR"].attrs["units"] = "dB"
                # _data["DWR"].attrs["long_name"] = "Ka-W DWR"

                # _data["DDV"] = self.ka.data["v"] - self.ww.data["v"]
                # _data["DDV"].attrs["units"] = "m/s"
                # _data["DDV"].attrs["long_name"] = "Ka-W DDV"
            return _data
        else:
            return self._data        

    def quicklook(self, variable: str | list[str], **kwargs) -> tuple[list[Figure] | None, list[Path] | None]:
        """
        Generate quicklook plots for the specified variables.
        Parameters:
        -----------
        variable : str or list of str
            The variable(s) to generate quicklook plots for. Must be a key or keys in self.data.
        **kwargs : dict
            Additional keyword arguments to pass to the plotting function.
        Returns:
        --------
        tuple[list[Figure] | None, list[Path] | None]
            A tuple containing a list of generated figures and a list of paths where the figures are saved.
            If no figures are generated, both lists will be None.
        """        
        nb_variables = [var for var in variable if var in self.data.keys()]

        #Create a fake NEBULA rpg instatnce to plot
        nb_rpg_class = rpg(self.ka.path)
        nb_rpg_class._data = self.data #replace ka data with nb data 
        nb_rpg_class._band = 'Dual'

        figures, paths = nb_rpg_class.quicklook(nb_variables, **kwargs)

        return figures, paths

    def plot_profile(
        self,
        variable: str | list[str],
        target_time: datetime | np.datetime64,
        range_limits: tuple[float, float],
        **kwargs,
    ) -> tuple[list[Figure] | None, list[Path] | None]:
        """
        Plots the profile of the specified variable(s) over the given time and range limits.
        Parameters:
        -----------
        target_time : datetime | np.datetime64
            The target time for which the profile is to be plotted.
        range_limits : tuple[float, float]
        variable : str | list[str]
            The variable(s) to be plotted. If None, all variables will be plotted.
            The range limits for the profile plot.
        **kwargs : dict
            Additional keyword arguments to customize the plot. Possible keys include:
            - savefig (bool): Whether to save the figure. Default is False.
            - ka_color (str): Color for the Ka-band plot. Default is "red".
            - w_color (str): Color for the W-band plot. Default is "blue".
            - output_dir (Path): Directory to save the figure if savefig is True. Default is current working directory.
            - dpi (int): Dots per inch for the saved figure. Default is 300.
        Returns:
        --------
        tuple[Figure, Path | None]
            The figure object and the path to the saved figure (if savefig is True), otherwise None.
        """
        
        _kwargs = kwargs.copy()        
        _kwargs["savefig"] = False
        _kwargs["color_list"] = [kwargs.get("ka_color", "red")]
        
        list_of_figs, list_of_paths = [], []

        if isinstance(variable, str):
            variables = [variable]
        elif isinstance(variable, list):
            variables = variable
        
        ka_ww_variables = [var for var in variables if (var in self.ka.data.keys() and var in self.ww.data.keys())]

        for variable_ in ka_ww_variables:
            fig1, _ = self.ka.plot_profile(
                target_times=target_time,
                range_limits=range_limits,
                variable=variable_,
                **_kwargs,
            )

            _kwargs["fig"] = fig1
            _kwargs["color_list"] = [kwargs.get("w_color", "blue")]
            
            fig, _ = self.ww.plot_profile(
                target_times=target_time,
                range_limits=range_limits,
                variable=variable_,
                **_kwargs,
            )

            ka_line, w_line = fig.findobj(Line2D)[0], fig.findobj(Line2D)[1]
            ka_line.set_label(f"Ka-band")
            w_line.set_label(f"W-band")

            ax = fig.get_axes()[0]
            ax.legend()

            if kwargs.get("savefig", False):
                filepath = (
                    kwargs.get("output_dir", Path.cwd())
                    / f"profile_{variable_}_{target_time:%Y%m%dT%H%M%S}.png"
                )
                fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
                list_of_figs.append(fig)
                list_of_paths.append(filepath) 
       
        #Plot dual products 
        #Create a fake NEBULA rpg instatnce to plot
        _kwargs = kwargs.copy()        
        _kwargs["savefig"] = False
        nb_rpg_class = rpg(self.ka.path)
        nb_rpg_class._data = self.data #replace ka data with nb data 
        nb_rpg_class._band = 'Dual'
        
        nb_variables = [var for var in variables if var in self.data.keys()]        
        for variable_ in nb_variables:
            fig, _ = nb_rpg_class.plot_profile(
                target_times=target_time,
                range_limits=range_limits,
                variable=variable_,
                **_kwargs,
            )
            
            if kwargs.get("savefig", False):
                filepath = (
                    kwargs.get("output_dir", Path.cwd())
                    / f"profile_{variable_}_{target_time:%Y%m%dT%H%M%S}.png"
                )
                fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
                list_of_figs.append(fig)
                list_of_paths.append(filepath) 
        if len(list_of_figs)==0 and len(list_of_paths) == 0:
            list_of_figs, list_of_paths = None, None
        
        return list_of_figs, list_of_paths

    def plot_timeseries(
        self,
        variable: str | list[str],
        target_range: float,
        time_limits: tuple[datetime | np.datetime64, datetime | np.datetime64] | None = None,
        **kwargs,
    ) -> tuple[list[Figure] | None, list[Path] | None]:
        """
        Plots a time series for the specified target range and variable.
        Parameters:
        -----------
        target_range : float
            The target range for which the time series is to be plotted.
        time_limits : tuple[datetime | np.datetime64, datetime | np.datetime64] | None, optional
            A tuple specifying the start and end times for the time series plot. If None, the full time range of the data is used.
        variable : str | None, optional
            The variable to be plotted. If None, a default variable is used.
        **kwargs : dict
            Additional keyword arguments to customize the plot. Common options include:
            - savefig (bool): Whether to save the figure to a file. Default is False.
            - ka_color (str): Color for the Ka-band plot. Default is "red".
            - w_color (str): Color for the W-band plot. Default is "blue".
            - output_dir (Path): Directory to save the figure if savefig is True. Default is the current working directory.
            - dpi (int): Dots per inch for the saved figure. Default is 300.
        Returns:
        --------
        tuple[Figure, Path | None]
            A tuple containing the figure object and the file path where the figure is saved (if applicable).
        """
        
        _kwargs = kwargs.copy()        
        _kwargs["savefig"] = False #Not directy save because modifications are required.
        _kwargs["color_list"] = [kwargs.get("ka_color", "red")]
        list_of_figs, list_of_paths = [], []

        if isinstance(variable, str):
            variables = [variable]
        elif isinstance(variable, list):
            variables = variable

        if  time_limits is None:
            time_limits = (parse_datetime(self.ka.data.time[0].values), parse_datetime(self.ka.data.time[-1].values))
        
        ka_ww_variables = [var for var in variables if (var in self.ka.data.keys() and var in self.ww.data.keys())]
                
        for variable in ka_ww_variables:
            fig1, _ = self.ka.plot_timeseries(
                target_ranges=target_range,
                time_limits=time_limits,
                variable=variable,
                **_kwargs,
            )

            _kwargs["fig"] = fig1
            _kwargs["color_list"] = [kwargs.get("w_color", "blue")]
            
            fig, _ = self.ww.plot_timeseries(
                target_ranges=target_range,
                time_limits=time_limits,
                variable=variable,
                **_kwargs,
            )
            
            ka_line, w_line = fig.findobj(Line2D)[0], fig.findobj(Line2D)[1]
            ka_line.set_label(f"Ka-band")
            w_line.set_label(f"W-band")

            ax = fig.get_axes()[0]
            variable_string = self.ka.data[variable].attrs['long_name'].replace('W-band','').replace("Ka-band",'')
            ax.set_ylabel(f"{variable_string}, [{self.ka.data[variable].attrs['units']}]")
            ax.legend()

            if kwargs.get("savefig", False):
                filepath = (
                    kwargs.get("output_dir", Path.cwd())
                    / f"timeseries_{variable}_{time_limits[0]:%Y%m%dT%H%M%S}_{time_limits[-1]:%Y%m%dT%H%M%S}_{target_range}.png"
                )
                fig.savefig(filepath, dpi=kwargs.get("dpi", 300))
                list_of_figs.append(fig)
                list_of_paths.append(filepath)

        #Plot dual products 
        #Create a fake NEBULA rpg instatnce to plot
        nb_rpg_class = rpg(self.ka.path)
        nb_rpg_class._data = self.data #replace ka data with nb data 
        nb_rpg_class._band = 'Dual'
        
        nb_variables = [var for var in variables if var in self.data.keys()]   
        _kwargs = kwargs.copy()
        _kwargs['savefig'] = False
        for variable_ in nb_variables:            
            fig_, _ = nb_rpg_class.plot_timeseries(
                target_ranges=target_range,
                time_limits=time_limits,
                variable=variable_,
                **_kwargs, #not directly used because no needs to savefig here.
            )

            if kwargs.get("savefig", False):
                filepath_ = (
                    kwargs.get("output_dir", Path.cwd())
                    / f"timeseries_{variable_}_{time_limits[0]:%Y%m%dT%H%M%S}_{time_limits[-1]:%Y%m%dT%H%M%S}_{target_range}.png"
                )                
                fig_.savefig(filepath_, dpi=kwargs.get("dpi", 300))
                list_of_figs.append(fig_)
                list_of_paths.append(filepath_) 
 
        if len(list_of_figs)==0 and len(list_of_paths) == 0:
            list_of_figs, list_of_paths = None, None
        
        return list_of_figs, list_of_paths


    def plot_spectrum(
        self, target_time: datetime | np.datetime64, target_range: float, **kwargs
    ) -> tuple[Figure, Path | None]:
        """
        Plots the spectrum for the given target time and range.
        Parameters:
        -----------
        target_time : datetime | np.datetime64
            The target time for which the spectrum is to be plotted.
        target_range : float
            The target range for which the spectrum is to be plotted.
        **kwargs : dict
            Additional keyword arguments to customize the plot. Possible keys include:
            - output_dir : Path, optional
                Directory where the plot will be saved if 'savefig' is True.
            - savefig : bool, optional
                If True, the plot will be saved to a file.
            - ka_color : str, optional
                Color for the Ka-band plot. Default is 'red'.
            - w_color : str, optional
                Color for the W-band plot. Default is 'blue'.
            - dpi : int, optional
                Dots per inch for the saved figure. Default is 300.
        Returns:
        --------
        tuple[Figure, Path | None]
            A tuple containing the figure object and the file path where the figure is saved (if 'savefig' is True).
        """
        _kwargs = kwargs.copy()
        _kwargs.pop("output_dir")
        _kwargs.pop("savefig")
        _kwargs["color"] = kwargs.get("ka_color", "red")
        fig, _ = self.ka.plot_spectrum(target_time, target_range, **_kwargs)

        _kwargs["fig"] = fig
        _kwargs["color"] = kwargs.get("w_color", "blue")
        fig, _ = self.ww.plot_spectrum(target_time, target_range, **_kwargs)
        ka_line, w_line = fig.findobj(Line2D)[0], fig.findobj(Line2D)[2]
        ka_line.set_label(f"Ka-band {ka_line.get_label()}")
        w_line.set_label(f"W-band {w_line.get_label()}")

        ax = fig.get_axes()[0]
        ax.legend()

        if kwargs.get("savefig", False):
            filepath = (
                kwargs.get("output_dir", Path.cwd())
                / f"{target_time:%Y%m%dT%H%M%S}_{target_range:.0f}_spectrum.png"
            )
            fig.savefig(filepath, dpi=kwargs.get("dpi", 300))

        return fig, filepath
