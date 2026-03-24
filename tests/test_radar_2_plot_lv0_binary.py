from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from dcrpy.rpg_binary import rpg
from dcrpy.retrieve.retrieve import retrieve_dBZe

ZEN_LV0 = Path(r"tests\data\RAW\nebula_ka\2024\03\13\240313_150001_P00_ZEN.LV0")
OUTPUT_DIR = Path(r"tests\figures")

def test_plot_testing():
    radar = rpg(ZEN_LV0)
    # fig, filepath = radar.plot_spectra_by_range(
    #     target_time=radar.raw.time[5].values,
    #     range_slice=[6000.0, 6050.0],
    #     **{"output_dir": Path(r"tests\figures")}
    # )

    target_time = 732034804
    target_time = datetime(2023, 3, 13, 15, 5, 0)
    target_range = 6000.0

    kwargs = {"color": "black", "velocity_limits": None}

    # chirp_number = int(
    #     data["chirp_number"].sel(range=target_range, method="nearest").values.item()
    # )

    data = radar.dataset.sel(
        time=target_time, range=target_range, method="nearest"
    ).copy()
    chirp_number = int(data["chirp_number"].item())

    velocity_vectors = data["velocity_vectors"].sel(chirp=chirp_number).values
    data = data.assign_coords(spectrum=velocity_vectors)

    # Convert to dBZe
    data["doppler_spectrum_v_dBZe"] = retrieve_dBZe(
        data["doppler_spectrum_v"], radar.band
    )
    data["doppler_spectrum_v_dBZe"].attrs = {
        "long_name": "Vertical power density",
        "units": "dB",
    }
    data["doppler_spectrum_h_dBZe"] = retrieve_dBZe(
        data["doppler_spectrum_h"], radar.band
    )
    data["doppler_spectrum_h_dBZe"].attrs = {
        "long_name": "Horizontal power density",
        "units": "dB",
    }

    data["doppler_spectrum_dBZe"] = retrieve_dBZe(data["doppler_spectrum"], radar.band)
    data["doppler_spectrum_dBZe"].attrs = {
        "long_name": "Horizontal power density",
        "units": "dB",
    }

    # data["doppler_spectrum_v_dBZe"].plot(
    #         ax=ax, color=kwargs.get("color", "black"), label='V'
    #     )
    fig, ax = plt.subplots(figsize=(10, 7))

    data["doppler_spectrum_h_dBZe"].plot(ax=ax, color="red", label="H")  # type: ignore
    data["doppler_spectrum_dBZe"].plot(ax=ax, color="green", label="T")  # type: ignore
    data["doppler_spectrum_v_dBZe"].plot(ax=ax, color="b", label="V")  # type: ignore
    ax.legend()

    fig.savefig("testing_spectrum_v-h-t.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    data["sZDR"].plot(ax=ax, color="b", label="sZDR")  # type: ignore
    ax.axhline(
        data["sZDRmax"].values.item(), ls="dashed", color="red", label="$sZDR_{max}$"
    )
    ax.set_ylim(-10, 2)
    ax.legend()
    fig.savefig("testing-sZDR.png")
    plt.close(fig)

    if kwargs.get("velocity_limits", None) is not None:
        ax.set_xlim(*kwargs.get("velocity_limits"))
    else:
        velocity_vectors = data["velocity_vectors"].sel(chirp=chirp_number).values
        ax.set_xlim(velocity_vectors.min(), velocity_vectors.max())

    ax.set_xlabel("Doppler velocity, [m/s]")
    ax.set_ylabel("Power density, [dB]")
    ax.set_title(f"Time: {str(target_time).split('.')[0]}, Range: {target_range}")

    # Add vertical lines at 0
    ax.axvline(x=0, color="black", linestyle="--")
    ax.legend(ncol=kwargs.get("ncol", 2), loc="upper right", fontsize=8)

    fig.savefig(OUTPUT_DIR / "testing.png")
    plt.close(fig)
    assert "doppler_spectrum_v" in data
    assert "doppler_spectrum_h" in data
    assert "doppler_spectrum" in data
