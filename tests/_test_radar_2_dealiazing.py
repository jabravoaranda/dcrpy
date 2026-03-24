from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from gfatpy.radar.rpg_binary import rpg
from gfatpy.radar.retrieve.retrieve import retrieve_dBZe

ZEN_LV0 = Path(r"tests\datos\RAW\nebula_w\2024\03\19\240319_150001_P00_ZEN.LV0")

def test_dealiaze_one_height():
    radar = rpg(ZEN_LV0)
    
    target_time=datetime(2024, 3, 19, 15, 5, 0)
    target_range = 1000.
    dataset = radar.dataset.sel(time=target_time, method="nearest").copy()    

    dataset = radar.dealiaze(dataset, target_ranges=target_range)