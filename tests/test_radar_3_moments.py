from pathlib import Path
import xarray as xr
from math import isclose
from dcrpy.rpg_binary import rpg

from dcrpy.retrieve.moments import ze_from_spectrum

ZEN = Path(r"tests\data\RAW\nebula_w\2024\03\13\240313_150001_P00_ZEN.LV0")

def test_ze():
    radar = rpg(ZEN)
    ze = ze_from_spectrum(radar.dataset, variable='doppler_spectrum')
    breakpoint()
    assert isinstance(ze, xr.DataArray)
    assert isclose(ze.values.max().item(), 13.371192932128906, rel_tol=0.05)