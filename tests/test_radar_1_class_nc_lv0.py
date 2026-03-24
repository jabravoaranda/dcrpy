from pathlib import Path
from pdb import set_trace

# from gfatpy.radar.rpg import rpg
from dcrpy.rpg_nc import rpg

ZEN_NC = Path(r"tests\data\PRODUCTS\nebula_ka\2024\03\13\240313_150001_P00_ZEN.LV0.nc")

def test_init_zen():
    radar = rpg(ZEN_NC)    
    assert radar.level == 0
    assert radar.type == "ZEN"
    assert str(radar.data.time[0].values) == '2024-03-13T15:00:01.000000000'