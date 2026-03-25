"""Retrieval and product-generation helpers for RPG radar datasets.

This module centralizes single-frequency retrieval formulas used throughout
``dcrpy``. The functions operate mostly on ``xarray`` objects and are designed
to be reused by both file-reader classes and downstream analysis code.

The module includes helpers for:

- converting linear radar variables to logarithmic units
- deriving polarimetric quantities such as PhiDP, KDP, and ZDR
- deriving vertical spectra and signal-to-noise ratios from spectral data
- augmenting level-0 and level-1 datasets with commonly used products
- downloading Cloudnet raw files through the Cloudnet API
"""

from typing import Any
import os

import numpy as np
import requests
import xarray as xr


def retrieve_dBZe(Ze: xr.DataArray, band: str) -> xr.DataArray:
    """Convert linear reflectivity to dBZe.

    Parameters
    ----------
    Ze : xarray.DataArray
        Reflectivity in linear units.
    band : str
        Radar band label used in metadata, typically ``"Ka"`` or ``"W"``.

    Returns
    -------
    xarray.DataArray
        Reflectivity in dBZe with invalid values masked to ``NaN``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        dBZe = xr.apply_ufunc(
            lambda x: 10 * np.log10(x),
            Ze,
            dask="parallelized",
        )
    dBZe = dBZe.where(np.isfinite(dBZe), np.nan)
    dBZe.attrs = {"long_name": f"{band}-band equivalent reflectivity", "units": "dBZe"}
    return dBZe


def retrieve_PhiDP(phiDP: xr.DataArray) -> xr.DataArray:
    """Convert differential phase from radians to degrees.

    Parameters
    ----------
    phiDP : xarray.DataArray
        Differential phase in radians.

    Returns
    -------
    xarray.DataArray
        Differential phase in degrees.

    Notes
    -----
    This helper currently performs only a unit conversion. The sign convention
    may still require scientific review, as noted in the original code.
    """
    phiDP_data = np.rad2deg(phiDP)
    phiDP_data = xr.DataArray(phiDP_data, coords=phiDP.coords, dims=phiDP.dims)
    phiDP_data.attrs = {
        "standard_name": "PhiDP",
        "long_name": "Differential phase shift",
        "units": "deg",
    }
    return phiDP_data


def retrieve_KDP(
    phiDP: xr.DataArray, moving_windows: tuple[int, int] = (1, 5)
) -> xr.DataArray:
    """Estimate specific differential phase from bulk PhiDP.

    Parameters
    ----------
    phiDP : xarray.DataArray
        Differential phase with dimensions including ``time`` and
        ``range_layers``.
    moving_windows : tuple[int, int], default: (1, 5)
        Time and range rolling-window lengths used before differentiating in
        range.

    Returns
    -------
    xarray.DataArray
        Specific differential phase in degrees per kilometer.

    Raises
    ------
    ValueError
        If the required ``time`` dimension is missing or too short.
    """
    range_resolution_array = np.diff(phiDP.range_layers)
    if "time" not in phiDP.dims:
        raise ValueError("No time dimension found")
    if len(phiDP.time) < 2:
        raise ValueError(
            "dimension time found to be less than 2. KDP calculation not possible."
        )

    time_window, range_window = moving_windows
    time_rolled_phiDP = phiDP.rolling(
        time=time_window, min_periods=1, center=True
    ).mean()
    range_time_rolled_phiDP = time_rolled_phiDP.rolling(
        range_layers=range_window, min_periods=1, center=True
    ).mean()
    specific_diff_phase_shift = range_time_rolled_phiDP.diff(dim="range_layers") / (
        2.0 * abs(range_resolution_array) * 1e-3
    )
    specific_diff_phase_shift = specific_diff_phase_shift.reindex(
        range_layers=phiDP.range_layers, method="nearest"
    )
    specific_diff_phase_shift = specific_diff_phase_shift.rename(
        "specific_diff_phase_shift"
    )
    specific_diff_phase_shift.attrs = {
        "long_name": "Specific differential phase shift",
        "units": "deg/km",
    }
    return specific_diff_phase_shift


def retrieve_spec_KDP(
    spec_phiDP: xr.DataArray, moving_windows: tuple[int, int] = (1, 5)
) -> xr.DataArray:
    """Estimate spectral specific differential phase from spectral PhiDP.

    Parameters
    ----------
    spec_phiDP : xarray.DataArray
        Spectral differential phase with dimensions including ``time`` and
        ``range_layers``.
    moving_windows : tuple[int, int], default: (1, 5)
        Time and range rolling-window lengths used before differentiating in
        range.

    Returns
    -------
    xarray.DataArray
        Spectral specific differential phase in degrees per kilometer.

    Raises
    ------
    ValueError
        If the required ``time`` dimension is missing or too short.
    """
    if "time" not in spec_phiDP.dims:
        raise ValueError("No time dimension found")
    if len(spec_phiDP.time) < 2:
        raise ValueError(
            "dimension time found to be less than 2. KDP calculation not possible."
        )

    time_window, range_window = moving_windows
    time_rolled_spec_phiDP = spec_phiDP.rolling(
        time=time_window, min_periods=1, center=True
    ).mean()
    range_time_rolled_spec_phiDP = time_rolled_spec_phiDP.rolling(
        range_layers=range_window, min_periods=1, center=True
    ).mean()
    specific_diff_phase_shift = range_time_rolled_spec_phiDP.diff(dim="range_layers") / (
        2.0 * abs(range_time_rolled_spec_phiDP.range_layers) * 1e-3
    )
    specific_diff_phase_shift = specific_diff_phase_shift.reindex(
        range_layers=spec_phiDP.range_layers, method="nearest"
    )
    specific_diff_phase_shift = specific_diff_phase_shift.rename(
        "specific_diff_phase_shift"
    )
    specific_diff_phase_shift.attrs = {
        "long_name": "Spectral specific differential phase shift",
        "units": "deg/km",
    }
    return specific_diff_phase_shift


def retrieve_wind_components(
    data: xr.Dataset, height: float | list[float] | np.ndarray[Any, np.dtype[np.float64]]
) -> xr.Dataset:
    """Placeholder for wind-component retrieval from Doppler velocities.

    Parameters
    ----------
    data : xarray.Dataset
        Radar dataset containing the information required for wind retrieval.
    height : float or list[float] or numpy.ndarray
        Height or heights at which wind components should be estimated.

    Returns
    -------
    xarray.Dataset
        A dataset containing wind components.

    Notes
    -----
    This function is intentionally not implemented yet.
    """
    ...


def add_all_products_from_LV1(raw: xr.Dataset, band: str) -> xr.Dataset:
    """Add a core set of derived products to an LV1 dataset.

    Parameters
    ----------
    raw : xarray.Dataset
        LV1 dataset containing at least ``Ze`` and ``differential_phase``.
    band : str
        Radar band label used when generating ``dBZe`` metadata.

    Returns
    -------
    xarray.Dataset
        Copy of ``raw`` with additional reflectivity and phase products.
    """
    data = raw.copy()
    data["dBZe"] = retrieve_dBZe(data["Ze"], band)
    data["differential_phase"] = retrieve_PhiDP(data["differential_phase"])
    data["specific_differential_phase"] = retrieve_KDP(data["differential_phase"])
    return data


def retrieve_NoiseDens(data: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Estimate horizontal and vertical spectral noise densities.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing chirp layout, integrated noise, and sample counts.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        Horizontal and vertical noise densities per range gate.
    """
    bins_per_chirp = np.diff(np.hstack((data["chirp_start_indices"], data["n_range_layers"])))
    noise_density_h = data["integrated_noise_h"] / np.repeat(
        data["n_samples_in_chirp"].values, bins_per_chirp
    )
    noise_density_h = noise_density_h.where(noise_density_h != 0, 1e-10)
    noise_density_v = data["integrated_noise"] / np.repeat(
        data["n_samples_in_chirp"].values, bins_per_chirp
    )
    noise_density_v = noise_density_v.where(noise_density_v != 0, 1e-10)
    return noise_density_h, noise_density_v


def retrieve_doppler_spectrum_v(data: xr.Dataset) -> xr.DataArray:
    """Derive the vertical Doppler spectrum from total, H, and covariance spectra.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing ``doppler_spectrum``, ``doppler_spectrum_h``, and
        ``covariance_spectrum_re``.

    Returns
    -------
    xarray.DataArray
        Vertical Doppler spectrum in linear units.
    """
    spectrum_v = (
        4 * data["doppler_spectrum"]
        - data["doppler_spectrum_h"]
        - 2 * data["covariance_spectrum_re"]
    )
    spectrum_v.name = "doppler_spectrum_v"
    spectrum_v.attrs = {
        "long_name": "Vertical power density",
        "units": "linear Ze",
    }
    return spectrum_v


def retrieve_spec_snr_h(data: xr.Dataset) -> xr.DataArray:
    """Compute horizontal spectral signal-to-noise ratio."""
    noise_density_h, _ = retrieve_NoiseDens(data)
    return data["doppler_spectrum_h"] / noise_density_h


def retrieve_spec_snr_v(data: xr.Dataset) -> xr.DataArray:
    """Compute vertical spectral signal-to-noise ratio."""
    if "doppler_spectrum_v" not in data:
        data["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(data)
    _, noise_density_v = retrieve_NoiseDens(data)
    return data["doppler_spectrum_v"] / noise_density_v


def retrieve_snr_v(data: xr.Dataset) -> xr.DataArray:
    """Compute bulk vertical signal-to-noise ratio from the spectral data."""
    if "doppler_spectrum_v" not in data:
        data["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(data)
    _, noise_density_v = retrieve_NoiseDens(data)
    power_noise_v = data["doppler_spectrum_v"].count(dim="spectrum") * noise_density_v
    power_signal_v = data["doppler_spectrum_v"].sum(dim="spectrum")
    return power_signal_v / power_noise_v


def retrieve_snr_h(data: xr.Dataset) -> xr.DataArray:
    """Compute bulk horizontal signal-to-noise ratio from the spectral data."""
    noise_density_h, _ = retrieve_NoiseDens(data)
    power_noise_h = data["doppler_spectrum_h"].count(dim="spectrum") * noise_density_h
    power_signal_h = data["doppler_spectrum_h"].sum(dim="spectrum")
    return power_signal_h / power_noise_h


def linear_to_dB(data: xr.DataArray) -> xr.DataArray:
    """Convert positive linear values to dB and mask invalid output.

    Parameters
    ----------
    data : xarray.DataArray
        Input data in linear units.

    Returns
    -------
    xarray.DataArray
        Data in dB with non-finite values masked to ``NaN``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 10 * np.log10(data)
    result = xr.DataArray(result, coords=data.coords, dims=data.dims)
    result = result.where(np.isfinite(result), np.nan)
    return result


def retrieve_spectral_PhiDp(data: xr.Dataset) -> xr.DataArray:
    """Compute spectral differential phase in degrees."""
    spec_PhiDP = np.arctan2(data.covariance_spectrum_im, data.covariance_spectrum_re)
    spec_PhiDP = np.rad2deg(spec_PhiDP)
    return spec_PhiDP


def retrieve_PhiDp(data: xr.Dataset) -> xr.DataArray:
    """Compute bulk differential phase in degrees from spectral covariance."""
    PhiDP = np.arctan2(
        data.covariance_spectrum_im.sum(dim="spectrum"),
        data.covariance_spectrum_re.sum(dim="spectrum"),
    )
    PhiDP = np.rad2deg(PhiDP)
    return PhiDP


def retrieve_spectral_ZDR(data: xr.Dataset) -> xr.DataArray:
    """Compute spectral differential reflectivity.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing horizontal and vertical Doppler spectra. The
        vertical spectrum is created on demand if absent.

    Returns
    -------
    xarray.DataArray
        Spectral ZDR in dB.
    """
    if "doppler_spectrum_v" not in data:
        data = data.copy()
        data["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(data)
    ratio = data["doppler_spectrum_h"] / data["doppler_spectrum_v"]
    ratio = ratio.where(ratio > 0, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = xr.DataArray(
            10 * np.log10(ratio),
            coords=data["doppler_spectrum_h"].coords,
            dims=data["doppler_spectrum_h"].dims,
        )
    ratio = ratio.where(np.isfinite(ratio), np.nan)
    ratio.attrs = {
        "long_name": "Spectral differential reflectivity",
        "units": "dB",
    }
    return ratio


def retrieve_ZDR(data: xr.Dataset) -> xr.DataArray:
    """Compute bulk differential reflectivity from integrated spectra.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing horizontal and vertical Doppler spectra.

    Returns
    -------
    xarray.DataArray
        Bulk ZDR in dB.
    """
    zh = data["doppler_spectrum_h"].sum(dim="spectrum")
    zv = data["doppler_spectrum_v"].sum(dim="spectrum")

    with np.errstate(divide="ignore", invalid="ignore"):
        zdr = 10 * np.log10(zh / zv)
    zdr = zdr.where(np.isfinite(zdr), np.nan)
    zdr.attrs = {
        "long_name": "Differential reflectivity",
        "units": "dB",
    }
    return zdr


def retrieve_spectral_rhv(data: xr.Dataset) -> xr.DataArray:
    """Compute the spectral correlation coefficient.

    Parameters
    ----------
    data : xarray.Dataset
        Dataset containing horizontal and vertical spectra plus real and
        imaginary covariance spectra.

    Returns
    -------
    xarray.DataArray
        Spectral correlation coefficient.

    Notes
    -----
    The implementation follows the same general structure as the RPG software
    formulation, using the complex covariance and noise-corrected H/V spectra.
    """
    Bhh = data["doppler_spectrum_h"] - data["integrated_noise_h"]
    Bvv = data["doppler_spectrum_v"] - data["integrated_noise"]
    Bhv_complex = data["covariance_spectrum_re"] + 1j * data["covariance_spectrum_im"]

    spec_rhv = np.abs(Bhv_complex) / np.sqrt(Bhh * Bvv)
    spec_rhv = xr.DataArray(
        spec_rhv,
        coords=data["doppler_spectrum_h"].coords,
        dims=data["doppler_spectrum_h"].dims,
    )
    spec_rhv.attrs = {
        "long_name": "Spectral correlation coefficient",
        "units": "unitless",
    }
    return spec_rhv


def add_all_products_from_LV0(raw: xr.Dataset) -> xr.Dataset:
    """Add a broad set of retrieval products to an LV0 spectral dataset.

    Parameters
    ----------
    raw : xarray.Dataset
        LV0 dataset containing spectral moments, covariance, chirp metadata,
        and noise information.

    Returns
    -------
    xarray.Dataset
        Copy of ``raw`` with additional polarimetric, SNR, and correlation
        products.

    Notes
    -----
    This helper is more expansive than :func:`add_all_products_from_LV1` and
    includes products that depend directly on spectral information.
    """
    data = raw.copy()
    chirp_number = xr.DataArray(
        np.zeros(data.range_layers.size), coords={"range_layers": data.range_layers}
    )
    for idx, start_chirp_ in enumerate(data["chirp_start_indices"].values):
        chirp_number.loc[{"range_layers": data.range_layers[start_chirp_:]}] = idx
    data["chirp_number"] = chirp_number
    data = data.assign_coords(chirp=np.sort(np.unique(chirp_number.values.astype(int))))

    data["spec_PhiDp"] = retrieve_spectral_PhiDp(data)
    data["PhiDp"] = retrieve_PhiDp(data)
    data["KDP"] = retrieve_KDP(data["PhiDp"], (1, 5))
    data["spec_KDP"] = retrieve_spec_KDP(data["spec_PhiDp"], (1, 5))
    data["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(data)

    _, NoiseDensV = retrieve_NoiseDens(data)

    data["snr_v"] = retrieve_snr_v(data)
    data["snr_h"] = retrieve_snr_h(data)
    data["sSNR_H"] = retrieve_spec_snr_h(data)
    data["sSNR_H_dB"] = 10 * np.log10(data["sSNR_H"])
    data["sSNR_V"] = retrieve_spec_snr_v(data)
    data["sSNR_V_dB"] = 10 * np.log10(data["sSNR_V"])
    data["spec_ZDR"] = retrieve_spectral_ZDR(data)
    data["ZDR"] = 10 * np.log10(data["ZDP"])

    snr_mask = (data["sSNR_V"] < 1000) | (data["sSNR_H"] < 1000)
    snr_mask = snr_mask | (data["doppler_spectrum"] == 0.0)
    data["srn_mask"] = snr_mask

    Bhh = data["doppler_spectrum_h"] - NoiseDensV
    Bvv = data["doppler_spectrum_v"] - NoiseDensV
    Bhv_complex = data["covariance_spectrum_re"] + 1j * data["covariance_spectrum_im"]

    data["spec_rhv"] = retrieve_spectral_rhv(data)
    data["rhv"] = np.abs(Bhv_complex.sum(dim="spectrum")) / np.sqrt(
        Bhh.sum(dim="spectrum") * Bvv.sum(dim="spectrum")
    )

    A = Bhh + Bvv - 2 * Bhv_complex.real
    B = Bhh + Bvv + 2 * Bhv_complex.real

    data["spec_sldr"] = 10 * np.log10(A / B)
    data["sldr"] = 10 * np.log10(A.sum(dim="spectrum") / B.sum(dim="spectrum"))
    return data


def download_cloudnet_data(
    instrument,
    site,
    date_from,
    suffix,
    pid,
    date_to=None,
    output_folder=".",
):
    """Download Cloudnet raw files via the Cloudnet API.

    Parameters
    ----------
    instrument : str
        Instrument name, for example ``"rpg-fmcw-94"``.
    site : str
        Cloudnet site name, for example ``"granada"``.
    date_from : str
        Start date in ``YYYY-MM-DD`` format.
    suffix : str
        Filename suffix filter sent to the API.
    pid : str
        Instrument PID sent to the API.
    date_to : str, optional
        End date in ``YYYY-MM-DD`` format. When omitted, ``date_from`` is used.
    output_folder : str, default: "."
        Destination directory for the downloaded files.

    Returns
    -------
    None
        Files are downloaded into ``output_folder``.
    """
    if date_to is None:
        date_to = date_from

    url = "https://cloudnet.fmi.fi/api/raw-files"
    payload = {
        "instrument": instrument,
        "site": site,
        "dateFrom": date_from,
        "dateTo": date_to,
        "filenameSuffix": suffix,
        "instrumentPid": pid,
    }
    metadata = requests.get(url, params=payload).json()

    os.makedirs(output_folder, exist_ok=True)

    for row in metadata:
        res = requests.get(row["downloadUrl"], stream=True)
        file_path = os.path.join(output_folder, row["filename"])
        with open(file_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
