import numpy as np
import xarray as xr


def ze_from_spectrum(data: xr.Dataset, variable: str = 'doppler_spectrum') -> xr.DataArray:
    """
    Calculate Ze (total power) from the Doppler spectrum.
    """
    return data[variable].sum(dim="spectrum")

def vm_from_spectrum(data: xr.Dataset) -> xr.DataArray:
    """
    Calculate mean Doppler velocity (Vm) from the Doppler spectrum.
    """
    ze = ze_from_spectrum(data)
    # velocity_vectors: (chirp, spectrum)
    # Map velocity_vectors to each range_layers using chirp_number
    velocity = xr.apply_ufunc(
        lambda chirp: data["velocity_vectors"].sel(chirp=chirp),
        data["chirp_number"],
        vectorize=True,
        input_core_dims=[["range"]],
        output_core_dims=[["spectrum"]],
        dask="parallelized",
        output_dtypes=[data["velocity_vectors"].dtype],
    )
    vm = (data["doppler_spectrum"] * velocity).sum(dim="spectrum") / ze
    return vm

def w_from_spectrum(data: xr.Dataset) -> xr.DataArray:
    """
    Calculate spectral width (W) from the Doppler spectrum.
    """
    ze = ze_from_spectrum(data)
    vm = vm_from_spectrum(data)
    velocity = xr.apply_ufunc(
        lambda chirp: data["velocity_vectors"].sel(chirp=chirp),
        data["chirp_number"],
        vectorize=True,
        input_core_dims=[["range"]],
        output_core_dims=[["spectrum"]],
        dask="parallelized",
        output_dtypes=[data["velocity_vectors"].dtype],
    )
    w2 = ((data["doppler_spectrum"] * (velocity - vm[..., None]) ** 2).sum(dim="spectrum")) / ze
    w = np.sqrt(w2)
    return w

def skew_from_spectrum(data: xr.Dataset) -> xr.DataArray:
    """
    Calculate skewness from the Doppler spectrum.
    """
    ze = ze_from_spectrum(data)
    vm = vm_from_spectrum(data)
    w = w_from_spectrum(data)
    velocity = xr.apply_ufunc(
        lambda chirp: data["velocity_vectors"].sel(chirp=chirp),
        data["chirp_number"],
        vectorize=True,
        input_core_dims=[["range_layers"]],
        output_core_dims=[["spectrum"]],
        dask="parallelized",
        output_dtypes=[data["velocity_vectors"].dtype],
    )
    skew = (
        (data["doppler_spectrum"] * ((velocity - vm[..., None]) / w[..., None]) ** 3).sum(dim="spectrum")
        / ze
    )
    return skew

def kurt_from_spectrum(data: xr.Dataset) -> xr.DataArray:
    """
    Calculate kurtosis from the Doppler spectrum.
    """
    ze = ze_from_spectrum(data)
    vm = vm_from_spectrum(data)
    w = w_from_spectrum(data)
    velocity = xr.apply_ufunc(
        lambda chirp: data["velocity_vectors"].sel(chirp=chirp),
        data["chirp_number"],
        vectorize=True,
        input_core_dims=[["range_layers"]],
        output_core_dims=[["spectrum"]],
        dask="parallelized",
        output_dtypes=[data["velocity_vectors"].dtype],
    )
    kurt = (
        (data["doppler_spectrum"] * ((velocity - vm[..., None]) / w[..., None]) ** 4).sum(dim="spectrum")
        / ze
    )
    return kurt
