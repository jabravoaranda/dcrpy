from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from dcrpy.retrieve.retrieve import (
    add_all_products_from_LV1,
    linear_to_dB,
    retrieve_KDP,
    retrieve_NoiseDens,
    retrieve_PhiDp,
    retrieve_ZDR,
    retrieve_dBZe,
    retrieve_doppler_spectrum_v,
    retrieve_spec_KDP,
    retrieve_spec_snr_h,
    retrieve_spec_snr_v,
    retrieve_snr_h,
    retrieve_snr_v,
    retrieve_spectral_PhiDp,
    retrieve_spectral_ZDR,
)
from dcrpy.rpg_binary import rpg


ZEN = Path(r"tests\data\RAW\nebula_w\2026\03\10\260310_020001_P00_ZEN.LV0")


@pytest.fixture()
def synthetic_retrieve_dataset() -> xr.Dataset:
    time = np.array(
        [
            np.datetime64("2024-03-13T00:00:00"),
            np.datetime64("2024-03-13T00:10:00"),
            np.datetime64("2024-03-13T00:20:00"),
        ]
    )
    range_layers = np.array([0.0, 1000.0, 2000.0])
    spectrum = np.array([0, 1])
    chirp = np.array([0, 1])

    doppler_spectrum_h = np.array(
        [
            [[2.0, 4.0], [3.0, 5.0], [4.0, 6.0]],
            [[2.5, 4.5], [3.5, 5.5], [4.5, 6.5]],
            [[3.0, 5.0], [4.0, 6.0], [5.0, 7.0]],
        ]
    )
    doppler_spectrum_v_expected = doppler_spectrum_h / 2.0 + 1.0
    covariance_spectrum_re = np.full_like(doppler_spectrum_h, 0.5)
    covariance_spectrum_im = np.full_like(doppler_spectrum_h, np.sqrt(3.0) / 2.0)
    doppler_spectrum = (
        doppler_spectrum_v_expected + doppler_spectrum_h + 2.0 * covariance_spectrum_re
    ) / 4.0

    dataset = xr.Dataset(
        coords={
            "time": time,
            "range_layers": range_layers,
            "spectrum": spectrum,
            "chirp": chirp,
        }
    )
    dataset["doppler_spectrum"] = (
        ("time", "range_layers", "spectrum"),
        doppler_spectrum,
    )
    dataset["doppler_spectrum_h"] = (
        ("time", "range_layers", "spectrum"),
        doppler_spectrum_h,
    )
    dataset["covariance_spectrum_re"] = (
        ("time", "range_layers", "spectrum"),
        covariance_spectrum_re,
    )
    dataset["covariance_spectrum_im"] = (
        ("time", "range_layers", "spectrum"),
        covariance_spectrum_im,
    )
    dataset["chirp_start_indices"] = (("chirp",), np.array([0, 2]))
    dataset["n_samples_in_chirp"] = (("chirp",), np.array([4.0, 8.0]))
    dataset["n_range_layers"] = xr.DataArray(3)
    dataset["integrated_noise_h"] = (("range_layers",), np.array([2.0, 2.0, 2.0]))
    dataset["integrated_noise"] = (("range_layers",), np.array([4.0, 4.0, 4.0]))
    return dataset


def test_retrieve_dBZe_matches_log10_and_masks_invalid():
    ze = xr.DataArray([1.0, 10.0, 100.0, 0.0, -1.0], dims=("gate",))

    dBZe = retrieve_dBZe(ze, "W")

    np.testing.assert_allclose(dBZe.values[:3], np.array([0.0, 10.0, 20.0]))
    assert np.isnan(dBZe.values[3])
    assert np.isnan(dBZe.values[4])
    assert dBZe.attrs["units"] == "dBZe"


def test_retrieve_doppler_spectrum_v_matches_definition(
    synthetic_retrieve_dataset: xr.Dataset,
):
    retrieved = retrieve_doppler_spectrum_v(synthetic_retrieve_dataset)
    expected = (
        4 * synthetic_retrieve_dataset["doppler_spectrum"]
        - synthetic_retrieve_dataset["doppler_spectrum_h"]
        - 2 * synthetic_retrieve_dataset["covariance_spectrum_re"]
    )

    xr.testing.assert_allclose(retrieved, expected)
    assert retrieved.name == "doppler_spectrum_v"


def test_retrieve_noise_density_matches_chirp_repetition(
    synthetic_retrieve_dataset: xr.Dataset,
):
    noise_density_h, noise_density_v = retrieve_NoiseDens(synthetic_retrieve_dataset)

    np.testing.assert_allclose(noise_density_h.values, np.array([0.5, 0.5, 0.25]))
    np.testing.assert_allclose(noise_density_v.values, np.array([1.0, 1.0, 0.5]))


def test_retrieve_spec_snr_h_and_v(
    synthetic_retrieve_dataset: xr.Dataset,
):
    dataset = synthetic_retrieve_dataset.copy()
    dataset["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(dataset)

    spec_snr_h = retrieve_spec_snr_h(dataset)
    spec_snr_v = retrieve_spec_snr_v(dataset)

    expected_h = dataset["doppler_spectrum_h"] / xr.DataArray(
        np.array([0.5, 0.5, 0.25]), coords={"range_layers": dataset["range_layers"]}
    )
    expected_v = dataset["doppler_spectrum_v"] / xr.DataArray(
        np.array([1.0, 1.0, 0.5]), coords={"range_layers": dataset["range_layers"]}
    )

    xr.testing.assert_allclose(spec_snr_h, expected_h)
    xr.testing.assert_allclose(spec_snr_v, expected_v)


def test_retrieve_snr_h_and_v(
    synthetic_retrieve_dataset: xr.Dataset,
):
    dataset = synthetic_retrieve_dataset.copy()
    dataset["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(dataset)

    snr_h = retrieve_snr_h(dataset)
    snr_v = retrieve_snr_v(dataset)

    expected_h = dataset["doppler_spectrum_h"].sum(dim="spectrum") / xr.DataArray(
        np.array([1.0, 1.0, 0.5]), coords={"range_layers": dataset["range_layers"]}
    )
    expected_v = dataset["doppler_spectrum_v"].sum(dim="spectrum") / xr.DataArray(
        np.array([2.0, 2.0, 1.0]), coords={"range_layers": dataset["range_layers"]}
    )

    xr.testing.assert_allclose(snr_h, expected_h)
    xr.testing.assert_allclose(snr_v, expected_v)


def test_linear_to_dB_masks_nonpositive_values():
    data = xr.DataArray([1.0, 10.0, 0.0, -1.0], dims=("gate",))

    result = linear_to_dB(data)

    np.testing.assert_allclose(result.values[:2], np.array([0.0, 10.0]))
    assert np.isnan(result.values[2])
    assert np.isnan(result.values[3])


def test_retrieve_spectral_phi_dp_and_phi_dp(
    synthetic_retrieve_dataset: xr.Dataset,
):
    spec_phi_dp = retrieve_spectral_PhiDp(synthetic_retrieve_dataset)
    phi_dp = retrieve_PhiDp(synthetic_retrieve_dataset)

    expected_angle = np.rad2deg(np.arctan2(np.sqrt(3.0) / 2.0, 0.5))
    np.testing.assert_allclose(spec_phi_dp.values, expected_angle)
    np.testing.assert_allclose(phi_dp.values, expected_angle)


def test_retrieve_spectral_zdr_and_zdr(
    synthetic_retrieve_dataset: xr.Dataset,
):
    dataset = synthetic_retrieve_dataset.copy()
    dataset["doppler_spectrum_v"] = retrieve_doppler_spectrum_v(dataset)

    spectral_zdr = retrieve_spectral_ZDR(dataset)
    zdr = retrieve_ZDR(dataset)

    expected_spectral = 10.0 * np.log10(
        dataset["doppler_spectrum_h"] / dataset["doppler_spectrum_v"]
    )
    expected_zdr = 10.0 * np.log10(
        dataset["doppler_spectrum_h"].sum(dim="spectrum")
        / dataset["doppler_spectrum_v"].sum(dim="spectrum")
    )

    xr.testing.assert_allclose(spectral_zdr, expected_spectral)
    xr.testing.assert_allclose(zdr, expected_zdr)
    assert zdr.attrs["units"] == "dB"


def test_retrieve_KDP_returns_expected_gradient():
    phi_dp = xr.DataArray(
        np.array([[0.0, 2.0, 4.0], [0.0, 2.0, 4.0], [0.0, 2.0, 4.0]]),
        coords={
            "time": np.array(
                [
                    np.datetime64("2024-03-13T00:00:00"),
                    np.datetime64("2024-03-13T00:10:00"),
                    np.datetime64("2024-03-13T00:20:00"),
                ]
            ),
            "range_layers": np.array([0.0, 1000.0, 2000.0]),
        },
        dims=("time", "range_layers"),
    )

    kdp = retrieve_KDP(phi_dp, moving_windows=(1, 1))

    np.testing.assert_allclose(kdp.values, np.ones_like(phi_dp.values))


def test_retrieve_spec_KDP_returns_expected_gradient():
    spec_phi_dp = xr.DataArray(
        np.array(
            [
                [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
                [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
                [[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
            ]
        ),
        coords={
            "time": np.array(
                [
                    np.datetime64("2024-03-13T00:00:00"),
                    np.datetime64("2024-03-13T00:10:00"),
                    np.datetime64("2024-03-13T00:20:00"),
                ]
            ),
            "range_layers": np.array([0.0, 1000.0, 2000.0]),
            "spectrum": np.array([0, 1]),
        },
        dims=("time", "range_layers", "spectrum"),
    )

    spec_kdp = retrieve_spec_KDP(spec_phi_dp, moving_windows=(1, 1))

    assert spec_kdp.shape == spec_phi_dp.shape
    assert np.isfinite(spec_kdp.values).all()
    np.testing.assert_allclose(spec_kdp.values[:, :2, :], 1.0)
    assert np.all(spec_kdp.values[:, 2, :] > 0.0)


def test_add_all_products_from_LV1_adds_dBZe_and_phase_products():
    raw = xr.Dataset(
        coords={
            "time": np.array(
                [
                    np.datetime64("2024-03-13T00:00:00"),
                    np.datetime64("2024-03-13T00:10:00"),
                    np.datetime64("2024-03-13T00:20:00"),
                ]
            ),
            "range_layers": np.array([0.0, 1000.0, 2000.0]),
        }
    )
    raw["Ze"] = (("time", "range_layers"), np.array([[1.0, 10.0, 100.0]] * 3))
    raw["differential_phase"] = (
        ("time", "range_layers"),
        np.deg2rad(np.array([[0.0, 2.0, 4.0]] * 3)),
    )

    products = add_all_products_from_LV1(raw, "W")

    assert "dBZe" in products
    assert "specific_differential_phase" in products
    assert "differential_phase" in products
    np.testing.assert_allclose(products["dBZe"].values[0], np.array([0.0, 10.0, 20.0]))
    np.testing.assert_allclose(products["differential_phase"].values[0], np.array([0.0, 2.0, 4.0]))
    assert products["specific_differential_phase"].shape == raw["differential_phase"].shape
    assert np.isfinite(products["specific_differential_phase"].values).all()
    assert np.all(products["specific_differential_phase"].values >= 0.0)


def test_retrieve_formula_matches_binary_dataset():
    radar = rpg(ZEN)
    dataset = radar.dataset

    expected = (
        4 * dataset["doppler_spectrum"]
        - dataset["doppler_spectrum_h"]
        - 2 * dataset["covariance_spectrum_re"]
    )

    xr.testing.assert_allclose(retrieve_doppler_spectrum_v(dataset), expected)
