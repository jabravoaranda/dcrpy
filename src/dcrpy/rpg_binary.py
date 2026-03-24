from pathlib import Path
import numpy as np
import xarray as xr
from rpgpy import read_rpg
from rpgpy.spcutil import scale_spectra

# from gfatpy.radar._dealiazing import dealiaze, dealiazeOneHeight


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
