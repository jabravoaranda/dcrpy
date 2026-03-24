from pathlib import Path

# from gfatpy.radar.rpg import rpg
from dcrpy.rpg_nc import rpg

ZEN_NC = Path(r"tests\data\PRODUCTS\nephele\2024\02\09\240209_015959_P00_ZEN.LV1.nc")

PPI_NC = Path(r"tests\data\PRODUCTS\nephele\2023\05\17\230517_084500_P00_PPI.LV1.nc")

RHI_NC = Path(r"tests\data\PRODUCTS\nephele\2024\02\09\240209_022515_P00_RHI.LV1.nc")

def test_init_zen(radar_files):
    radar = rpg(ZEN_NC)    
    assert radar.level == 1
    assert radar.type == "ZEN"
    assert str(radar.data.time[0].values) == '2024-02-09T01:59:59.000000000'
    assert 'dBZe' in radar.data    
    assert 'differential_phase' in radar.data
    assert 'specific_differential_phase' in radar.data


def test_init_ppi(radar_files):
    radar = rpg(PPI_NC)

    assert radar.level == 1
    assert radar.type == "PPI"
    assert str(radar.data.time[0].values) == '2023-05-17T08:45:00.000000000'
    assert 'dBZe' in radar.data    
    assert 'differential_phase' in radar.data
    assert 'specific_differential_phase' in radar.data


def test_init_rhi(radar_files):

    radar = rpg(RHI_NC)

    assert radar.level == 1
    assert radar.type == "RHI"
    assert str(radar.data.time[0].values) == '2024-02-09T02:25:15.000000000'
    assert 'dBZe' in radar.data    
    assert 'differential_phase' in radar.data
    assert 'specific_differential_phase' in radar.data

