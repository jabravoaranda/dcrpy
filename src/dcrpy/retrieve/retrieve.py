from typing import Any
import xarray as xr
import numpy as np
import os
import requests

def retrieve_dBZe(Ze: xr.DataArray, band: str) -> xr.DataArray:
    with np.errstate(divide='ignore', invalid='ignore'):
        dBZe = xr.apply_ufunc(
            lambda x: 10 * np.log10(x),
            Ze,
            dask='parallelized',
        )  # convert to dBZ, 10log10(x) = 10 * log10(x)
    dBZe = dBZe.where(np.isfinite(dBZe), np.nan)
    dBZe.attrs = {"long_name": f"{band}-band equivalent reflectivity", "units": "dBZe"}
    return dBZe

def retrieve_PhiDP(phiDP: xr.DataArray) -> xr.DataArray:
    phiDP_data = np.rad2deg(
        phiDP
    )  # convert to deg, add -1 because convention is other way around 
       #(now the phase shift gets negative, we want it to get positive with range_layers...) 
       # TODO: check with Alexander if that makes sense!!
    
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
    # time window: timewindow*timeres gives the amount of seconds over which will be averaged
    # calculate KDP from phidp directly
    range_resolution_array = np.diff(phiDP.range_layers)
    if not "time" in phiDP.dims:
        raise ValueError("No time dimension found")
    if len(phiDP.time) < 2:
        raise ValueError(
            "dimension time found to be less than 2. KDP calculation not possible."
        )
    
    time_window, range_window = moving_windows
    time_rolled_phiDP = phiDP.rolling(
        time=time_window, min_periods=1, center=True
    ).mean()  # moving window average in time
    range_time_rolled_phiDP = time_rolled_phiDP.rolling(
        range_layers=range_window, min_periods=1, center=True
    ).mean()  # moving window average in range_layers
    specific_diff_phase_shift = range_time_rolled_phiDP.diff(dim="range_layers") / (
        2.0 * abs(range_resolution_array) * 1e-3
    )  # in order to get °/km we need to multiply with 1e-3
    specific_diff_phase_shift = specific_diff_phase_shift.reindex(range_layers=phiDP.range_layers, method="nearest")
    specific_diff_phase_shift = specific_diff_phase_shift.rename(
        "specific_diff_phase_shift"
    )
    specific_diff_phase_shift.attrs = {
        "long_name": "Specific differential phase shift",
        "units": "°/km",
    }
    return specific_diff_phase_shift

def retrieve_spec_KDP( 
    spec_phiDP: xr.DataArray, moving_windows: tuple[int, int] = (1, 5)
) -> xr.DataArray:
    # time window: timewindow*timeres gives the amount of seconds over which will be averaged
    # calculate KDP from phidp directly
    range_resolution_array = np.diff(spec_phiDP.range_layers)
    if not "time" in spec_phiDP.dims:
        raise ValueError("No time dimension found")
    if len(spec_phiDP.time) < 2:
        raise ValueError(
            "dimension time found to be less than 2. KDP calculation not possible."
        )
    
    time_window, range_window = moving_windows
    time_rolled_spec_phiDP = spec_phiDP.rolling(
        time=time_window, min_periods=1, center=True
    ).mean()  # moving window average in time
    range_time_rolled_spec_phiDP = time_rolled_spec_phiDP.rolling(
        range_layers=range_window, min_periods=1, center=True
    ).mean()  # moving window average in range_layers
    specific_diff_phase_shift = range_time_rolled_spec_phiDP.diff(dim="range_layers") / (
        2.0 * abs(range_time_rolled_spec_phiDP.range_layers) * 1e-3
    )  # in order to get °/km we need to multiply with 1e-3
    specific_diff_phase_shift = specific_diff_phase_shift.reindex(range_layers=spec_phiDP.range_layers, method="nearest")
    specific_diff_phase_shift = specific_diff_phase_shift.rename(
        "specific_diff_phase_shift"
    )
    specific_diff_phase_shift.attrs = {
        "long_name": "Spectral specific differential phase shift",
        "units": "°/km",
    }
    return specific_diff_phase_shift

def retrieve_wind_components(data: xr.Dataset, height: float | list[float] | np.ndarray[Any, np.dtype[np.float64]]) -> xr.Dataset:
    # Calculate the wind components from the Doppler velocities
    #TODO: Implement the actual wind component retrieval logic
    ...

def add_all_products_from_LV1(raw: xr.Dataset, band: str) -> xr.Dataset:
    # Add products becomes too big.
    # How far can we go with LV1 data? Is it need to use LV0 from the scratch?
    # Implement here the functions made by Chris
    # It may be insteresting to compare the netcdf from the RPGpy and the one from RPG software
    # There are products that can be only calculated from the spectral data
    data = raw.copy()
    data["dBZe"] = retrieve_dBZe(data["Ze"], band)    
    data["differential_phase"] = retrieve_PhiDP(data["differential_phase"])    
    data["specific_differential_phase"] = retrieve_KDP(data["differential_phase"])
    return data

def retrieve_NoiseDens(data: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    bins_per_chirp = np.diff(np.hstack((data["chirp_start_indices"], data["n_range_layers"])))
    noise_density_h = data['integrated_noise_h'] / np.repeat(data["n_samples_in_chirp"].values, bins_per_chirp)
    noise_density_h = noise_density_h.where(noise_density_h != 0, 1e-10)  # Avoid division by zero
    noise_density_v = data['integrated_noise'] / np.repeat(data["n_samples_in_chirp"].values, bins_per_chirp)
    noise_density_v = noise_density_v.where(noise_density_v != 0, 1e-10)  # Avoid division by zero
    return noise_density_h, noise_density_v

def retrieve_doppler_spectrum_v(data: xr.Dataset) -> xr.DataArray:
    spectrum_v = (
        4 * data['doppler_spectrum']
        - data['doppler_spectrum_h']
        - 2 * data['covariance_spectrum_re']
    )
    spectrum_v.name = 'doppler_spectrum_v'
    spectrum_v.attrs = {
        "long_name": "Vertical power density",
        "units": "linear Ze",
    }
    return spectrum_v

def retrieve_spec_snr_h(data: xr.Dataset) -> xr.DataArray:
    noise_density_h, _ = retrieve_NoiseDens(data)
    return data['doppler_spectrum_h'] / noise_density_h

def retrieve_spec_snr_v(data: xr.Dataset) -> xr.DataArray:
    if 'doppler_spectrum_v' not in data:
        data['doppler_spectrum_v'] = retrieve_doppler_spectrum_v(data)
    _, noise_density_v = retrieve_NoiseDens(data)
    
    return data['doppler_spectrum_v'] / noise_density_v

def retrieve_snr_v(data: xr.Dataset) -> xr.DataArray:
    if 'doppler_spectrum_v' not in data:
        data['doppler_spectrum_v'] = retrieve_doppler_spectrum_v(data)
    _, noise_density_v = retrieve_NoiseDens(data)
    power_noise_v = data['doppler_spectrum_v'].count(dim='spectrum') * noise_density_v
    power_signal_v = data['doppler_spectrum_v'].sum(dim='spectrum') 
    return power_signal_v / power_noise_v

def retrieve_snr_h(data: xr.Dataset) -> xr.DataArray:
    noise_density_h, _ = retrieve_NoiseDens(data)
    power_noise_h  = data['doppler_spectrum_h'].count(dim='spectrum') * noise_density_h
    power_signal_h = data['doppler_spectrum_h'].sum(dim='spectrum')
    return power_signal_h / power_noise_h

def linear_to_dB(data: xr.DataArray) -> xr.DataArray:
    with np.errstate(divide='ignore', invalid='ignore'):
        result = 10 * np.log10(data)
    result = xr.DataArray(result, coords=data.coords, dims=data.dims)
    result = result.where(np.isfinite(result), np.nan)  # Replace -inf and inf with NaN

    return result

def retrieve_spectral_PhiDp(data: xr.Dataset) -> xr.DataArray:
    # Calculate the spectral differential phase shift
    spec_PhiDP = np.arctan2(data.covariance_spectrum_im, data.covariance_spectrum_re)
    spec_PhiDP = np.rad2deg(spec_PhiDP)  # Convert to degrees
    return spec_PhiDP

def retrieve_PhiDp(data: xr.Dataset) -> xr.DataArray:
    # Calculate the differential phase shift
    PhiDP = np.arctan2(data.covariance_spectrum_im.sum(dim="spectrum"), 
                       data.covariance_spectrum_re.sum(dim="spectrum"))
    PhiDP = np.rad2deg(PhiDP)  # Convert to degrees
    return PhiDP

def retrieve_spectral_ZDR(data: xr.Dataset) -> xr.DataArray:
    # Calculate the differential reflectivity
    if 'doppler_spectrum_v' not in data:
        data = data.copy()
        data['doppler_spectrum_v'] = retrieve_doppler_spectrum_v(data)
    ratio = data['doppler_spectrum_h'] / data['doppler_spectrum_v']
    ratio = ratio.where(ratio > 0, np.nan)  # Replace invalid values with NaN

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = xr.DataArray(
            10 * np.log10(ratio),
            coords=data['doppler_spectrum_h'].coords,
            dims=data['doppler_spectrum_h'].dims,
        )
    ratio = ratio.where(np.isfinite(ratio), np.nan)
    ratio.attrs = {
        "long_name": "Spectral differential reflectivity",
        "units": "dB",
    }

    return ratio

def retrieve_ZDR(data: xr.Dataset) -> xr.DataArray:
    # Calculate the differential reflectivity
    zh = data['doppler_spectrum_h'].sum(dim='spectrum')
    zv = data['doppler_spectrum_v'].sum(dim='spectrum')

    ZDR = xr.DataArray(10 * np.log10(zh / zv), 
                       coords=data['doppler_spectrum_h'].coords, 
                       dims=data['doppler_spectrum_h'].dims)

    return ZDR

def retrieve_spectral_rhv(data: xr.Dataset) -> xr.DataArray:
    """
    Calculate the spectral correlation coefficient (spectral rhv).

    Args:
        data (xr.Dataset): The radar dataset containing doppler_spectrum_h, doppler_spectrum_v,
                            covariance_spectrum_re, and covariance_spectrum_im.

    Note:
        RPGpy does:
        rhv = np.abs(data["ReVHSpec"] + complex(imag=1) * data["ImVHSpec"]) / np.sqrt((spec_V + noise_v_per_bin) * (data["HSpec"] + noise_h_per_bin)

    Returns:
        xr.DataArray: The spectral rhv.
    """

    Bhh = data['doppler_spectrum_h'] - data['integrated_noise_h']
    Bvv = data['doppler_spectrum_v'] - data['integrated_noise']
    Bhv_complex = data['covariance_spectrum_re'] + 1j * data['covariance_spectrum_im']

    spec_rhv = np.abs(Bhv_complex) / np.sqrt(Bhh * Bvv)

    spec_rhv = xr.DataArray(spec_rhv, coords=data['doppler_spectrum_h'].coords, dims=data['doppler_spectrum_h'].dims)

    spec_rhv.attrs = {
        "long_name": "Spectral correlation coefficient",
        "units": "unitless",
    }
    
    return spec_rhv

def add_all_products_from_LV0(raw: xr.Dataset) -> xr.Dataset:
    data = raw.copy()       
    chirp_number = xr.DataArray(np.zeros(data.range_layers.size), coords={'range_layers': data.range_layers})   
    for idx, start_chirp_ in enumerate(data['chirp_start_indices'].values):
        chirp_number.loc[{'range_layers': data.range_layers[start_chirp_:]}] = idx
    data['chirp_number'] = chirp_number
    data = data.assign_coords(chirp = np.sort(np.unique(chirp_number.values.astype(int)))) 
    
    # spectral PhiDp
    data['spec_PhiDp'] = retrieve_spectral_PhiDp(data)
    data['PhiDp']    = retrieve_PhiDp(data)
    data['KDP']      = retrieve_KDP(data['PhiDp'], (1, 5))
    data['spec_KDP'] = retrieve_spec_KDP(data['spec_PhiDp'], (1, 5))
    
    data['doppler_spectrum_v'] = retrieve_doppler_spectrum_v(data)

    _, NoiseDensV = retrieve_NoiseDens(data)

    data['snr_v'] = retrieve_snr_v(data)
    data['snr_h'] = retrieve_snr_h(data)
    data['sSNR_H'] = retrieve_spec_snr_h(data)
    data["sSNR_H_dB"] = 10*np.log10(data["sSNR_H"])
    data['sSNR_V'] = retrieve_spec_snr_v(data)
    data["sSNR_V_dB"] = 10*np.log10(data["sSNR_V"])
    data['spec_ZDR'] = retrieve_spectral_ZDR(data)
    data['ZDR'] = 10*np.log10(data['ZDP']) # ZDR in logscale dB
    
    snr_mask = (data['sSNR_V'] < 1000) | (data['sSNR_H'] < 1000)
    snr_mask = snr_mask | (data['doppler_spectrum'] == 0.0)
    data['srn_mask'] = snr_mask 

    Bhh = data['doppler_spectrum_h'] - NoiseDensV
    Bvv = data['doppler_spectrum_v'] - NoiseDensV
    Bhv_complex = data['covariance_spectrum_re'] + 1j*data['covariance_spectrum_im']

    data['spec_rhv'] = retrieve_spectral_rhv(data)
    data['rhv']      = np.abs(Bhv_complex.sum(dim="spectrum")) / np.sqrt(Bhh.sum(dim="spectrum") * Bvv.sum(dim="spectrum"))

    A = Bhh + Bvv - 2*Bhv_complex.real
    B = Bhh + Bvv + 2*Bhv_complex.real

    data['spec_sldr'] = 10 * np.log10(A / B)
    data['sldr'] = 10 * np.log10(A.sum(dim="spectrum") / B.sum(dim="spectrum"))
    return data

def download_cloudnet_data(instrument, site, date_from, suffix, pid, date_to=None, output_folder='.'):
    """
    Downloads Cloudnet data for the specified instrument, site, and date range_layers.

    Args:
        instrument (str): The instrument name (e.g., 'rpg-fmcw-94').
        site (str): The site name (e.g., 'granada').
        date_from (str): The start date in 'YYYY-MM-DD' format.
        date_to (str, optional): The end date in 'YYYY-MM-DD' format. Defaults to None.
        output_folder (str, optional): The folder path where the files will be saved. Defaults to '.'.
    """
    # If no end date is provided, use the start date as the end date
    if date_to is None:
        date_to = date_from

    url = 'https://cloudnet.fmi.fi/api/raw-files'
    payload = {
        'instrument': instrument,
        'site': site,
        'dateFrom': date_from,
        'dateTo': date_to,
        'filenameSuffix':suffix,
        'instrumentPid': pid,
    }
    metadata = requests.get(url, params=payload).json()

    # Ensure the output folder exists; create it if not.
    os.makedirs(output_folder, exist_ok=True)

    for row in metadata:
        res = requests.get(row['downloadUrl'], stream=True)
        
        # Create the full file path by joining the output folder and the filename
        file_path = os.path.join(output_folder, row['filename'])
        
        with open(file_path, 'wb') as f:
            for chunk in res.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive new chunks
                    f.write(chunk)
