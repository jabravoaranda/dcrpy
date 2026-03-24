import pandas as pd
from pathlib import Path

from matplotlib.figure import Figure

from dcrpy.rpg_binary import rpg

ZEN = Path(r"tests\data\RAW\nebula_w\2024\03\13\240313_150001_P00_ZEN.LV0")

def test_plot_spectrum():
    radar = rpg(ZEN)
    fig, filepath = radar.plot_spectrum(
        target_range=500.,
        target_time=radar.dataset.time[0].values,
        variable_to_plot='doppler_spectrum_h',
        **{"output_dir": Path(r"tests\figures")}
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()

def test_plot_spectra_by_range():
    radar = rpg(ZEN)
    fig, filepath = radar.plot_spectra_by_range(
        target_time=radar.dataset.time[10].values,
        range_slice=[500.0, 1000.0],
        **{"output_dir": Path(r"tests\figures")}
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()


def test_plot_2d_spectrum():
    radar = rpg(ZEN)

    fig, filepath = radar.plot_2D_spectrum(
        variable_to_plot='doppler_spectrum_h',
        target_time=radar.dataset.time[1].values,
        range_limits=None,
        **{"output_dir": Path(r"tests\figures")}
    )

    assert isinstance(fig, Figure)
    assert filepath is not None
    assert filepath.exists()

def test_plot_spectra_by_time():
    radar = rpg(ZEN)    
    fig, filepath = radar.plot_spectra_by_time(
        target_range=8000,
        time_slice=(radar.dataset.time[0].values, radar.dataset.time[5].values),
        **{"output_dir": Path(r"tests\figures")}
    )

    assert isinstance(fig, Figure)
    assert filepath.exists()


def test_plot_profile(): 
    radar = rpg(ZEN)    
    times = pd.date_range( start=radar.dataset.time.values[0], end=radar.dataset.time.values[100], freq="10min" ).to_pydatetime().tolist()
    breakpoint()
    radar.plot_profile(
        target_times=times,
        range_limits=(0., 12000.0),
        variable="dBZe",
        output_dir=Path(r"tests\figures"),
        savefig=True,
        **{'ncol': 1}
    )
    assert (Path(r"tests\figures") / "240209_015959_P00_ZEN.LV1_dBZe_profile_20240209T0200_20240209T0220.png").exists()


# def test_plot_testing(radar_files):
#     radar = rpg(ZEN_NC)
#     # fig, filepath = radar.plot_spectra_by_range(
#     #     target_time=radar.raw.time[5].values,
#     #     range_slice=[6000.0, 6050.0],
#     #     **{"output_dir": Path(r"tests\figures")}
#     # )
#     from dcrpy.retrieve.retrieve import retrieve_dBZe
#     import matplotlib.pyplot as plt
#     import numpy as np

#     kwargs = {'color': 'black', 'velocity_limits': None}
#     target_time=radar.raw.time[5].values
#     target_range = 6000.

#     data = radar.data.copy()

#     fig, ax = plt.subplots(figsize=(10, 7))
#     chirp_number = int(
#         data["chirp_number"].sel(range=target_range, method="nearest").values.item()
#     )
#     sdata = data.sel(time=target_time, range=target_range, method="nearest")
#     velocity_vectors = sdata["velocity_vectors"].sel(chirp=chirp_number).values
#     sdata = sdata.assign_coords(spectrum=velocity_vectors)

#     # Convert to dBZe
#     sdata["doppler_spectrum_v_dBZe"] = retrieve_dBZe(sdata["doppler_spectrum_v"], radar.band)
#     sdata["doppler_spectrum_v_dBZe"].attrs = {
#         "long_name": "Vertical power density",
#         "units": "dB",
#     }
#     sdata["doppler_spectrum_h_dBZe"] = retrieve_dBZe(sdata["doppler_spectrum_h"], radar.band)
#     sdata["doppler_spectrum_h_dBZe"].attrs = {
#         "long_name": "Horizontal power density",
#         "units": "dB",
#     }

#     sdata["doppler_spectrum_dBZe"] = retrieve_dBZe(sdata["doppler_spectrum"], radar.band)
#     sdata["doppler_spectrum_dBZe"].attrs = {
#         "long_name": "Horizontal power density",
#         "units": "dB",
#     }

#     # sdata["doppler_spectrum_v_dBZe"].plot(
#     #         ax=ax, color=kwargs.get("color", "black"), label='V'
#     #     )
#     sdata["doppler_spectrum_h_dBZe"].plot(
#             ax=ax, color=kwargs.get("color", "red"), label='H'
#         )
#     sdata["doppler_spectrum_dBZe"].plot(
#             ax=ax, color=kwargs.get("color", "green"), label='T'
#         )    
#     sdata['covariance_spectrum_re'].plot(
#             ax=ax, color=kwargs.get("color", "red"), label='T'
#         )
#     fig.savefig('testing.png')

#     breakpoint()
#     if kwargs.get("velocity_limits", None) is not None:
#         ax.set_xlim(*kwargs.get("velocity_limits"))
#     else:
#         nyquist_velocity = data["nyquist_velocity"].sel(chirp=chirp_number).values
#         ax.set_xlim(-nyquist_velocity, nyquist_velocity)

#     ax.set_xlabel("Doppler velocity, [m/s]")
#     ax.set_ylabel("Power density, [dB]")
#     ax.set_title(f"Time: {str(target_time).split('.')[0]}, Range: {target_range}")

#     # Add vertical lines at 0
#     ax.axvline(x=0, color="black", linestyle="--")
#     ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)

#     fig.savefig('testing.png')
#     breakpoint()

    # assert isinstance(fig, Figure)
    # assert filepath.exists()

