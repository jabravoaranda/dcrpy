import re
from pathlib import Path
from typing import Any
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd
from loguru import logger

""" MODULE For General Radar Utilities
"""

def str_to_datetime(date_str: str) -> dt.datetime:
    """

    Parameters
    ----------
    date_str: str
        date in string format (see possible formats below)

    Returns
    -------

    """
    assert isinstance(date_str, str), "date_str must be String Type"

    formats = [
        (r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}", "%Y%m%dT%H%M%S"),
        (r"\d{4}\d{2}\d{2}_\d{2}\d{2}\d{2}", "%Y%m%d_%H%M%S"),
        (r"\d{4}\d{2}\d{2}T\d{2}\d{2}", "%Y%m%dT%H%M"),
        (r"\d{4}\d{2}\d{2}_\d{2}\d{2}", "%Y%m%d_%H%M"),
        (r"\d{4}\d{2}\d{2}T\d{2}", "%Y%m%dT%H"),
        (r"\d{4}\d{2}\d{2}_\d{2}", "%Y%m%d_%H"),
        (r"\d{4}\d{2}\d{2}", "%Y%m%d"),
        (r"\d{4}\d{2}", "%Y%m"),
        (r"\d{4}", "%Y"),
    ]

    i = 0
    match = False
    date_format = ""
    date_dt = None
    while not match:
        if i < len(formats):
            candidate = re.search(formats[i][0], date_str)
            if candidate is not None:
                date_format = formats[i][1]
                match = True
            else:
                i += 1
        else:
            match = True
    if date_format is not None:
        try:
            date_dt = dt.datetime.strptime(date_str, date_format)
        except Exception as e:
            print(f"{date_str} has more complex format than found ({date_format})")
            raise NotImplementedError(e.with_traceback)
    else:
        raise RuntimeError(f"Cannot understand the format of {date_str}")

    return date_dt


def datetime_np2dt(numpy_date):
    """
    Converts a numpy datetime64 object to a python datetime object
    Input:
      date - a np.datetime64 object
    Output:
      DATE - a python datetime object
    """
    # timestamp = ((date - np.datetime64('1970-01-01T00:00:00'))
    #             / np.timedelta64(1, 's'))

    try:
        timestamp = pd.Timestamp(numpy_date).to_pydatetime()
    except Exception as e:
        timestamp = None
        logger.error(str(e))
        raise e
    return timestamp


def parse_datetime(
    date: dt.datetime | dt.date | str | np.datetime64 | pd.DatetimeIndex | pd.Timestamp,
) -> dt.datetime:
    """The general way to cast strigs or datetimes to datetime python type

    Args:
        date (dt.datetime | str): This could be one of the supported formats.
        The recommended one is ISO8601.

    Returns:
        dt.datetime: Python datetime format.
    """

    if isinstance(date, dt.datetime):
        return date
    elif isinstance(date, dt.date):
        return dt.datetime(date.year, date.month, date.day)
    elif isinstance(date, str):
        try:
            return dt.datetime.fromisoformat(date)
        except ValueError:
            return str_to_datetime(date)
    elif isinstance(date, np.datetime64):
        return datetime_np2dt(date)
    elif isinstance(date, pd.DatetimeIndex):
        return date.to_pydatetime()[0]
    elif isinstance(date, pd.Timestamp):
        return date.to_pydatetime()
    else:
        raise ValueError(f"{date} is not a valid date")


def check_is_netcdf(path) -> Path:
    """Check if path is a netcdf file

    Args:

        - path (_type_): Radar file path.

    Raises:
        - FileNotFoundError: File does not exist.
        - ValueError: File is not a netcdf file.

    Returns:
        - Path: Path to netcdf file.
    """    
    
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File {path} does not exist.")
    if path.suffix.lower() != ".nc":
        raise ValueError(f"File {path} is not a netcdf file.")
    return path

def enhance_rpgpy_dataset(
    path: Path, time_parser: str = "seconds since 2001-01-01 00:00:00 UTC"
) -> xr.Dataset:
    data = xr.open_dataset(path)
    if "Time" in data:
        data = data.rename({"Time": "time"})
    if "range_layers" in data:
        data["range"] = data["range_layers"]
    data.time.attrs["standard_name"] = "time"
    data.time.attrs["long_name"] = "time"
    data.time.attrs["units"] = time_parser
    data = xr.decode_cf(data)    
    return data


def ppi_to_cartessian(
    ranges: np.ndarray[np.float64] | xr.DataArray,
    azimuth: np.ndarray[np.float64] | xr.DataArray,
    elevation: np.ndarray[np.float64] | xr.DataArray,
) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Convert PPI coordinates to cartessian coordinates

    Args:
        ranges (np.ndarray[np.float64] | xr.DataArray): Range of the radar.
        azimuth (np.ndarray[np.float64] | xr.DataArray): Azimuth angle.
        elevation (np.ndarray[np.float64] | xr.DataArray): Elevation angle.

    Raises:
        ValueError: Elevation angle is not constant.

    Returns:
        tuple[np.ndarray[np.float64], np.ndarray[np.float64]]: x, y coordinates in cartessian plane [units as `ranges`].
    """    
    
    if isinstance(ranges, xr.DataArray):
        ranges = ranges.values
    if isinstance(azimuth, xr.DataArray):
        azimuth = azimuth.values
    if isinstance(elevation, xr.DataArray):
        elevation = elevation.values

    # Check elevation angle is constant
    if not np.allclose(elevation, elevation[0], atol=0.1):
        raise ValueError("Elevation angle is not constant.")
    elevation = elevation[0]
    rho = ranges * np.cos(np.deg2rad(elevation))
    x = rho[:, np.newaxis]  * np.sin(np.deg2rad(azimuth))
    y = rho[:, np.newaxis]  * np.cos(np.deg2rad(azimuth))
    return x, y

def rhi_to_cartessian(
    ranges: np.ndarray[np.float64] | xr.DataArray,    
    azimuth: np.ndarray[np.float64] | xr.DataArray,
    elevation: np.ndarray[np.float64] | xr.DataArray,
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Convert RHI coordinates to cartessian coordinates

    Args:
        ranges (np.ndarray[np.float64] | xr.DataArray): Range of the radar.
        azimuth (np.ndarray[np.float64] | xr.DataArray): Azimuth angle.
        elevation (np.ndarray[np.float64] | xr.DataArray): Elevation angle.

    Raises:
        ValueError: Azimuth angle is not constant.

    Returns:
        tuple[np.ndarray[np.float64], np.ndarray[np.float64]]: x, y coordinates in cartessian plane [units as `ranges`].
    """    

    if isinstance(ranges, xr.DataArray):
        ranges = ranges.values
    if isinstance(azimuth, xr.DataArray):
        azimuth = azimuth.values
    if isinstance(elevation, xr.DataArray):
        elevation = elevation.values

    # Check elevation angle is constant    
    if not np.allclose(azimuth, azimuth[0], atol=0.1):
        raise ValueError("Azimuth angle is not constant.")
    
    x = ranges[:, np.newaxis]  * np.cos(np.deg2rad(elevation))
    y = ranges[:, np.newaxis]  * np.sin(np.deg2rad(elevation))

    return x, y

def spherical_to_cartessian(
    ranges: np.ndarray[np.float64] | xr.DataArray,    
    azimuth: np.ndarray[np.float64] | xr.DataArray,
    elevation: np.ndarray[np.float64] | xr.DataArray,
    ) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """Convert RHI coordinates to cartessian coordinates

    Args:
        ranges (np.ndarray[np.float64] | xr.DataArray): Range of the radar.
        azimuth (np.ndarray[np.float64] | xr.DataArray): Azimuth angle.
        elevation (np.ndarray[np.float64] | xr.DataArray): Elevation angle.

    Raises:
        ValueError: Azimuth angle is not constant.

    Returns:
        tuple[np.ndarray[np.float64], np.ndarray[np.float64]]: x, y coordinates in cartessian plane [units as `ranges`].
    """    

    if isinstance(ranges, xr.DataArray):
        ranges = ranges.values
    if isinstance(azimuth, xr.DataArray):
        azimuth = azimuth.values
    if isinstance(elevation, xr.DataArray):
        elevation = elevation.values

    # # Check elevation angle is constant    
    # if not np.allclose(azimuth, azimuth[0], atol=0.1):
    #     raise ValueError("Azimuth angle is not constant.")
    
    # Convert azimth to speherical coordinates convention
    theta = 90 - elevation
    phi = convert_azimuth_y_to_x(azimuth, direction='clockwise')

    x = ranges[:, np.newaxis]  * np.sin(np.deg2rad(theta)) * np.cos(np.deg2rad(phi))
    y = ranges[:, np.newaxis]  * np.sin(np.deg2rad(theta)) * np.sin(np.deg2rad(phi))
    z = ranges[:, np.newaxis]  * np.cos(np.deg2rad(theta))

    return ranges, theta, phi, x, y, z

def convert_azimuth_y_to_x(angles_y, direction='clockwise'):
    """
    Convert an array of azimuth angles from being with respect to the y-axis to the x-axis.

    Parameters:
    angles_y (np.ndarray): Array of azimuth angles in degrees with respect to the y-axis.
    direction (str): Direction of measurement ('clockwise' or 'anticlockwise').

    Returns:
    np.ndarray: Array of azimuth angles in degrees with respect to the x-axis.
    """
    if direction == 'clockwise':
        angles_x = (90 - angles_y) % 360
    elif direction == 'anticlockwise':
        angles_x = (angles_y - 90) % 360
    else:
        raise ValueError("Direction must be 'clockwise' or 'anticlockwise'")
    
    return angles_x

def histogram_intersection(histogram1: np.ndarray[Any, np.float64], histogram2: np.ndarray[Any, np.float64], bins: np.ndarray[Any, np.float64]) -> float:
    """Compute the histogram intersection between two histograms.

    Args:

        - histogram1 (_type_): Histogram 1.
        - histogram2 (_type_): Histogram 2.
        - bins (_type_): Bins of the histograms.

    Returns:

        - float: 
    """    
    bins = np.diff(bins)
    sm = 0
    for i in range(len(bins)):
        sm += min(bins[i] * histogram1[i], bins[i] * histogram2[i])
    return sm

def mergeChirps_LV0(data):
# this function merges the chirps of the dataset. I am doing this, because the non-polarimetric X and Ka-Band don't have a chirps and therefore we only have one fixed range-resolution
# input: data as xarray-dataset, if you set spec=True, the spectral data will be merged.
    try:
        chirpNum = data.ChirpNum.values[0]
    except:
        chirpNum = data.ChirpNum.values

    try:
        if len(data.DoppLen) > 3:
            maxVel = data.MaxVel.values[0]
            doppLen = data.DoppLen.values[0]
        else:
            maxVel = data.MaxVel.values
            doppLen = data.DoppLen.values
    except:
        maxVel = data.MaxVel.values
        doppLen = data.DoppLen.values

    dv_vec = np.empty(chirpNum)
    maxRange = [max(data.C1Range), max(data.C2Range), max(data.C3Range)]
    for chirp in range(chirpNum):
        ChRange = 'C{chirp}Range'.format(chirp=chirp+1)
        ChHSpec = 'C{chirp}HSpec'.format(chirp=chirp+1)
        ChVSpec = 'C{chirp}VSpec'.format(chirp=chirp+1)
        ChReVHSpec = 'C{chirp}ReVHSpec'.format(chirp=chirp+1)
        ChHNoisePow = 'C{chirp}HNoisePow'.format(chirp=chirp+1)
        ChVNoisePow = 'C{chirp}VNoisePow'.format(chirp=chirp+1)
        ChVel = 'C{chirp}Vel'.format(chirp=chirp+1)
        
        #- calculate noise density:
        NoiseDensV = data[ChVNoisePow]/doppLen[chirp]
        NoiseDensH = data[ChHNoisePow]/doppLen[chirp]
        #- now we need to decompose the VSpec, because this is not actually the vert. Spectrum but saved as a composite of H and V
        data[ChVSpec] = 4*data[ChVSpec] - data[ChHSpec] - 2*data[ChReVHSpec]
        #- there was a software mistake, so the noise of ReHV was not stored correctly. Therefore Alexander suggested to use s SNR threshold of 10dB, otherwise the data will be masked. For this we need to calculate SNR: SNR = signal power/noise power. In order to calculate the correct values, we need to mask all values below -90 dBZ.
        specThreshold = 10**(-90/10)
        data[ChHSpec] = data[ChHSpec].where(data[ChHSpec]>specThreshold,np.NaN)
        data[ChVSpec] = data[ChVSpec].where(data[ChVSpec]>specThreshold,np.NaN)
        NoisePowH = data[ChHSpec].count(dim=ChVel)*NoiseDensH
        NoisePowV = data[ChVSpec].count(dim=ChVel)*NoiseDensV
        SignalPowH = data[ChHSpec].sum(dim=ChVel)
        SignalPowV = data[ChVSpec].sum(dim=ChVel)
        data['SNR_H'] = SignalPowH / NoisePowH
        data['SNR_V'] = SignalPowV / NoisePowV
        #-- now also make it spectral (so spectral ZDR against SNR in respective bin (we see also some high ZDR values at the left (fast) edge, which should not be there)) and then the data should be masked with sSNR > 10dB aswell
        data['sSNR_H'] = data[ChHSpec] / NoiseDensH
        data['sSNR_V'] = data[ChVSpec] / NoiseDensH 
        #-- it is the easiest to calculate Z,ZDR,ZDP at this stage, because once the chirps are merged, it is difficult to sort out the dv needed to integrate over the spectrum...
        data['ZH'] = data[ChHSpec].sum(dim=ChVel)
        data['ZV'] = data[ChVSpec].sum(dim=ChVel)
        data['ZDR'] = 10*np.log10(data['ZH'])-10*np.log10(data['ZV'])
        data['ZDP'] = data['ZH']-data['ZV']
        #- because the different chirps have a different Doppler resolution, we also need to regrid along that axis:
        velData = np.linspace(-maxVel[chirp], maxVel[chirp], doppLen[chirp], dtype=np.float32) # in the dataformat, the dopplervelocity was not assigned yet, but rather it is stored as maxVel and doppLen which you then need to manually assign to the doppler Vel coordinate
        dv_diff = np.diff(velData) # since we regrid along the doppler axis, we need to divide the regridded doppler spectra by dv
        #print(dv_diff)
        dv_vec[chirp] = dv_diff[0]
        data = data.assign({ChVel:velData})
        velRef = np.linspace(-maxVel[0], maxVel[0], doppLen[0], dtype=np.float32)# just use the Doppler velocity from the smallest chirp
        data = data.reindex({ChVel:velRef}, method = 'nearest', tolerance = 0.05) # regrid
        data[ChVSpec] = data[ChVSpec]/dv_vec[chirp]
        data[ChHSpec] = data[ChHSpec]/dv_vec[chirp]
        data[ChReVHSpec] = data[ChReVHSpec]/dv_vec[chirp]
        #- now we can rename the variables to without the chirps and then we merge the datasets along the range coordinate
        dataCh = data[[ChVSpec,ChHSpec,ChHNoisePow,ChVNoisePow,'ZH','ZV','ZDR','ZDP','SNR_H','SNR_V','sSNR_H','sSNR_V']]
        dataCh = dataCh.rename({ChRange:'range',ChVSpec:'VSpec',ChHSpec:'HSpec',ChHNoisePow:'HNoisePow',ChVNoisePow:'VNoisePow',ChVel:'Vel'})
        if chirp==0:
            finalData = dataCh
        else:
            finalData = xr.concat([finalData,dataCh],dim='range')
    dv =  xr.DataArray(dv_vec,dims=('Chirp'))
    temp = xr.Dataset({'Azm':data.Azm,
                    'Elv':data.Elv,
                    'RangeRes':data.RangeRes,
                    #'delRange':delRange,
                    'dv':dv,
                    'maxVel':maxVel,
                    'doppLen':doppLen,
                    'maxRange':maxRange
                    })
    
    finalData = xr.merge([finalData, temp])
    return finalData 