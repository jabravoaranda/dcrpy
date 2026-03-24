import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from dcrpy.nebula import nebula

MEAS_TYPE = "ZEN"
BANDS = ["ka", "w"]
YEAR, MONTH, DAY = 2024, 3, 13
HOUR_STR = "150001"

MAIN_DIR = Path(r"tests/datos")
OUTPUT_DIR = MAIN_DIR / "PRODUCTS" / "nebula"
KA_DIR = (
    MAIN_DIR
    / "PRODUCTS"
    / "nebula_ka"
    / f"{YEAR:04d}"
    / f"{MONTH:02d}"
    / f"{DAY:02d}"
) 
WW_DIR = (
    MAIN_DIR
    / "PRODUCTS"
    / "nebula_w"
    / f"{YEAR:04d}"
    / f"{MONTH:02d}"
    / f"{DAY:02d}"
)

KA_OUTPUT_DIR = MAIN_DIR / "PRODUCTS" / "nebula_ka"
WW_OUTPUT_DIR = MAIN_DIR / "PRODUCTS" / "nebula_w"
QUICLOOK_DIR = MAIN_DIR / "PRODUCTS" / "nebula" / "quicklooks"
ka_path = KA_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV1.nc"
ww_path = WW_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV1.nc"

# def test_quicklook(radar_files):
#     NEBULA_LV1 = nebula(ka_path, ww_path)
#     assert NEBULA_LV1.level == 1
#     assert NEBULA_LV1.type == 'ZEN'
#     assert len(NEBULA_LV1.paths) == 2

#     if not QUICLOOK_DIR.exists():
#         QUICLOOK_DIR.mkdir(parents=True, exist_ok=True)

#     kwargs = { 'output_dir': QUICLOOK_DIR, 'savefig': True }
#     NEBULA_LV1.quicklook(variable= ['dBZe', 'DWR', 'DDV'], **kwargs)

#     ka_figure = QUICLOOK_DIR / NEBULA_LV1.ka.path.name.replace(".nc", f"_{NEBULA_LV1.ka.band}-dBZe.png")
#     ww_figure = QUICLOOK_DIR / NEBULA_LV1.ww.path.name.replace(".nc", f"_{NEBULA_LV1.ww.band}-dBZe.png")
#     dwr_figure = QUICLOOK_DIR / NEBULA_LV1.ka.path.name.replace(".nc", f"_Dual-DWR.png")
#     ddv_figure = QUICLOOK_DIR / NEBULA_LV1.ka.path.name.replace(".nc", f"_Dual-DDV.png")
    
#     assert ka_figure.exists()
#     assert ww_figure.exists()
#     assert dwr_figure.exists()
#     assert ddv_figure.exists()


def test_plot_profile(radar_files):
    NEBULA_LV1 = nebula(ka_path, ww_path)

    target_time = datetime(2024, 3, 13, 15, 0, 0, 0)
    range_limits = (5000.0, 8000.0)

    fig_, filepath = NEBULA_LV1.plot_profile(
        target_time=target_time,
        range_limits=range_limits,
        variable=["dBZe", "DWR"],
        output_dir=QUICLOOK_DIR.parent,
        savefig=True,
    )
    
    assert filepath is not None
    assert filepath[0].exists()
    plt.close('all')

def test_plot_timeseries(radar_files):    
    NEBULA_LV1 = nebula(ka_path, ww_path)

    target_range = 8000.
    fig_, filepath = NEBULA_LV1.plot_timeseries(
        target_range=target_range,
        variable=["dBZe", "DWR"],
        output_dir=QUICLOOK_DIR.parent,
        savefig=True,
    )
    assert filepath is not None
    assert filepath[0].exists()
    plt.close('all')

def test_plot_ww(radar_files):
    ka_path = KA_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV0.nc"
    ww_path = WW_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV0.nc"

    NEBULA_LV0 = nebula(ka_path, ww_path)

    target_time = datetime(2024, 3, 13, 15, 0, 0, 0)
    target_range = 6000.
    kwargs = { 'output_dir': WW_OUTPUT_DIR, 'savefig': True }
    if not kwargs['output_dir'].exists():
        kwargs['output_dir'].mkdir(parents=True, exist_ok=True)

    fig_, filepath = NEBULA_LV0.ww.plot_spectrum(target_time=target_time, target_range=target_range, **kwargs)

    assert isinstance(filepath, Path)
    assert filepath.exists()
    plt.close(fig_)

    time_slice = (datetime(2024, 3, 13, 15, 0, 0, 0), datetime(2024, 3, 13, 15, 1, 0, 0))
    target_range = 6000.
    fig_, filepath = NEBULA_LV0.ww.plot_spectra_by_time(time_slice=time_slice, target_range=target_range, **kwargs)

    assert isinstance(filepath, Path)
    assert filepath.exists()
    plt.close(fig_)

    #Plot 2D spectrum
    target_time = datetime(2024, 3, 13, 15, 0, 0, 0)
    range_limits = (5000., 8000.)
    kwargs['colorbar_label'] = 'Power spectral density (dB)'
    fig_, filepath = NEBULA_LV0.ww.plot_2D_spectrum(target_time=target_time, range_limits=range_limits, **kwargs)
    assert isinstance(filepath, Path)
    assert filepath.exists()
    plt.close(fig_)

def test_plot_ka(radar_files):
    ka_path = KA_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV0.nc"
    ww_path = WW_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV0.nc"
    NEBULA_LV0 = nebula(ka_path, ww_path)

    target_time = datetime(2024, 3, 13, 15, 0, 0, 0)
    target_range = 6000.
    kwargs = { 'output_dir': KA_OUTPUT_DIR, 'savefig': True }
    if not kwargs['output_dir'].exists():
        kwargs['output_dir'].mkdir(parents=True, exist_ok=True)

    fig_, filepath = NEBULA_LV0.ka.plot_spectrum(target_time=target_time, target_range=target_range, **kwargs)

    assert isinstance(filepath, Path)
    assert filepath.exists()
    plt.close(fig_)

    time_slice = (datetime(2024, 3, 13, 15, 0, 0, 0), datetime(2024, 3, 13, 15, 1, 0, 0))
    target_range = 6000.
    fig_, filepath = NEBULA_LV0.ka.plot_spectra_by_time(time_slice=time_slice, target_range=target_range, **kwargs)

    assert isinstance(filepath, Path)
    assert filepath.exists()
    plt.close(fig_)

    #Plot 2D spectrum
    target_time = datetime(2024, 3, 13, 15, 0, 0, 0)
    range_limits = (5000., 8000.)
    kwargs['colorbar_label'] = 'Power spectral density (dB)'
    fig_, filepath = NEBULA_LV0.ka.plot_2D_spectrum(target_time=target_time, range_limits=range_limits, **kwargs)

    assert isinstance(filepath, Path)
    assert filepath.exists()
    plt.close(fig_)

def test_plot_ka_ww(radar_files):
    ka_path = KA_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV0.nc"
    ww_path = WW_DIR / f"{YEAR-2000:02d}{MONTH:02d}{DAY:02d}_{HOUR_STR}_P00_{MEAS_TYPE}.LV0.nc"
    NEBULA_LV0 = nebula(ka_path, ww_path)
    target_time = datetime(2024, 3, 13, 15, 0, 0, 0)
    target_range = 6000.
    for target_range in np.arange(6000., 8000, 250):
        kwargs = { 'output_dir': OUTPUT_DIR, 'savefig': True, 'velocity_limits': (-1.5,0.) }
        if not kwargs['output_dir'].exists():
            kwargs['output_dir'].mkdir(parents=True, exist_ok=True)

        fig_, filepath = NEBULA_LV0.plot_spectrum(target_time=target_time, target_range=target_range, **kwargs)
        plt.close(fig_)
    assert isinstance(filepath, Path)
    assert filepath.exists()
    
