from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from dcrpy.dual_rpg import dual_rpg
from dcrpy.rpg_binary import rpg


KA_LV0 = Path(r"tests\data\RAW\nebula_ka\2026\03\10\260310_020001_P00_ZEN.LV0")
W_LV0 = Path(r"tests\data\RAW\nebula_w\2026\03\10\260310_020001_P00_ZEN.LV0")


@pytest.fixture(scope="module")
def dual_radar() -> dual_rpg:
    return dual_rpg(KA_LV0, W_LV0)


def test_dual_rpg_assigns_ka_and_w_roles_independently_of_input_order():
    radar = dual_rpg(W_LV0, KA_LV0)

    assert radar.ka.band == "Ka"
    assert radar.w.band == "W"
    assert radar.paths["Ka"] == KA_LV0
    assert radar.paths["W"] == W_LV0


def test_dual_rpg_requires_one_ka_and_one_w_band():
    with pytest.raises(ValueError, match="one Ka-band radar and one W-band"):
        dual_rpg(W_LV0, W_LV0)


def test_dual_rpg_dataset_contains_prefixed_single_band_products(
    dual_radar: dual_rpg,
):
    dataset = dual_radar.dataset

    assert "ka_dBZe" in dataset
    assert "w_dBZe" in dataset
    assert "DFR_Ka_W" in dataset
    assert "ka_doppler_spectrum_dB" in dataset
    assert "w_doppler_spectrum_dB" in dataset
    assert dataset.attrs["reference_band"] == "W"
    assert dataset["DFR_Ka_W"].attrs["units"] == "dB"


def test_dual_rpg_dataset_uses_reference_band_grid(dual_radar: dual_rpg):
    dataset = dual_radar.dataset
    w_dataset = dual_radar.w.dataset

    xr.testing.assert_equal(dataset["time"], w_dataset["time"])
    xr.testing.assert_equal(dataset["range"], w_dataset["range"])


def test_dual_rpg_dfr_matches_ka_minus_w_reflectivity(dual_radar: dual_rpg):
    dataset = dual_radar.dataset

    xr.testing.assert_allclose(
        dataset["DFR_Ka_W"],
        dataset["ka_dBZe"] - dataset["w_dBZe"],
    )


def test_dual_rpg_spectral_dfr_line_returns_reference_grid_product(dual_radar: dual_rpg):
    dataset = dual_radar.dataset
    target_time = dataset.time.values[0]
    target_range = float(dataset.range.values[len(dataset.range) // 2])

    line = dual_radar._spectral_dfr_line(target_time, target_range)

    assert line.dims == ("spectrum",)
    assert line.sizes["spectrum"] == dataset.sizes["spectrum"]
    assert line.attrs["units"] == "dB"
    assert np.isfinite(line.values).any()


def test_dual_rpg_accepts_existing_single_band_objects():
    ka_radar = rpg(KA_LV0)
    w_radar = rpg(W_LV0)

    radar = dual_rpg(ka_radar, w_radar, reference_band="Ka")

    assert radar.reference_band == "Ka"
    assert radar.reference_radar is ka_radar
    assert radar.secondary_radar is w_radar
