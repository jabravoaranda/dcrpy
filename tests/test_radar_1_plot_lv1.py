import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from gfatpy.radar.rpg_nc import rpg

ZEN_NC = Path(r"tests\data\PRODUCTS\nephele\2024\02\09\240209_015959_P00_ZEN.LV1.nc")
PPI_NC = Path(r"tests\data\PRODUCTS\nephele\2023\05\17\230517_084500_P00_PPI.LV1.nc")
RHI_NC = Path(r"tests\data\PRODUCTS\nephele\2024\02\09\240209_022515_P00_RHI.LV1.nc")
OUTPUT_DIR = Path(r"tests\figures")

def test_ppi_quicklook(radar_files):
    radar = rpg(PPI_NC)
    radar.quicklook(
        variable=[
            "dBZe",
            "v",
            "width",
            "ldr",
            "specific_differential_phase",
        ],
        dpi=400,
        savefig=True,
        output_dir=OUTPUT_DIR,
    )
    assert Path(OUTPUT_DIR / "230517_084500_P00_PPI.LV1_W-dBZe.png").exists()


def test_rhi_quicklook(radar_files):
    radar = rpg(RHI_NC)
    radar._data = radar.data.sel(time=slice('2024-02-09T02:27:50.000000000','2024-02-09T02:30:35.000000000'))
    
    radar.quicklook(
        variable=[
            "dBZe",
            "v",
            "width",
            "ldr",
            "specific_differential_phase",
        ],
        dpi=400,
        savefig=True,
        output_dir=OUTPUT_DIR,
    )

    assert Path(OUTPUT_DIR / "240209_022515_P00_RHI.LV1_W-dBZe.png").exists()

def test_zen_quicklook(radar_files):
    radar = rpg(ZEN_NC)
    radar.quicklook(
        variable=[
            "dBZe",
            "v",
            "width",
            "skewness",
            "kurtosis",
            "ldr",
            "differential_phase",
            "specific_differential_phase",
        ],
        dpi=400,
        savefig=True,
        output_dir=OUTPUT_DIR,
    )
    assert Path(OUTPUT_DIR / "240209_015959_P00_ZEN.LV1_W-dBZe.png").exists()

def test_plot_profile(radar_files): 
    radar = rpg(ZEN_NC)
    radar.plot_profile(
        target_times=np.arange(datetime(2024, 2, 9, 2, 0, 0, 0), datetime(2024, 2, 9, 2, 30, 0, 0), timedelta(minutes=10)).tolist(),
        range_limits=(0., 12000.0),
        variable="dBZe",
        output_dir=OUTPUT_DIR,
        savefig=True,
        **{'ncol': 1}
    )
    assert Path(OUTPUT_DIR / "240209_015959_P00_ZEN.LV1_dBZe_profile_20240209T0200_20240209T0220.png").exists()

def test_plot_timeseries(radar_files): 
    radar = rpg(ZEN_NC)
    radar.plot_timeseries(        
        target_ranges=(4000., 4500.0),
        variable="dBZe",
        output_dir=OUTPUT_DIR,
        savefig=True,
        **{'ncol': 1}
    )    
    assert Path(OUTPUT_DIR / "240209_015959_P00_ZEN.LV1_dBZe_timeseries_20240209T0159_20240209T0225_4000.0_4500.0.png").exists()

def test_plot_timeseries(radar_files): 
    radar = rpg(ZEN_NC)
    radar.plot_timeseries(        
        target_ranges=(4000., 4500.0),
        variable="v",
        output_dir=OUTPUT_DIR,
        savefig=True,
        **{'ncol': 1}
    )    
    assert Path(OUTPUT_DIR / "240209_015959_P00_ZEN.LV1_v_timeseries_20240209T0159_20240209T0225_4000.0_4500.0.png").exists()
