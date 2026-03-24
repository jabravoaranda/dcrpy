from pathlib import Path

from matplotlib.figure import Figure

from dcrpy.rpg_nc import rpg

ZEN_NC = Path(r"tests\data\RAW\nephele\2021\09\13\210913_110000-110006_P05_ZEN.LV0.nc")


def test_plot_spectrum(radar_files):
    radar = rpg(ZEN_NC)
    fig, filepath = radar.plot_spectrum(
        target_range=8000,
        target_time=radar.raw.time[0].values,
        **{"output_dir": Path(r"tests\figures")}
    )

    assert isinstance(fig, Figure)
    assert filepath.exists()


def test_plot_spectra_by_time(radar_files):
    radar = rpg(ZEN_NC)
    fig, filepath = radar.plot_spectra_by_time(
        target_range=8000,
        time_slice=(radar.raw.time[0].values, radar.raw.time[5].values),
        **{"output_dir": Path(r"tests\figures")}
    )

    assert isinstance(fig, Figure)
    assert filepath.exists()


def test_plot_spectra_by_range(radar_files):
    radar = rpg(ZEN_NC)
    fig, filepath = radar.plot_spectra_by_range(
        target_time=radar.raw.time[5].values,
        range_slice=[6000.0, 6050.0],
        **{"output_dir": Path(r"tests\figures")}
    )
    assert isinstance(fig, Figure)
    assert filepath.exists()


def test_plot_2d_spectrum(radar_files):
    radar = rpg(ZEN_NC)
    fig, filepath = radar.plot_2D_spectrum(
        target_time=radar.raw.time[0].values,
        range_limits=(2000, 10000),
        vmin=-0.1,
        vmax=0.1,
        **{"output_dir": Path(r"tests\figures")}
    )

    assert isinstance(fig, Figure)
    assert filepath.exists()

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

