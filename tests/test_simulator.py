"""Sanity checks for the simulator.  Run:  python -m pytest tests"""
import numpy as np
import otfs


def test_noiseless_perfect_csi_has_no_errors():
    # Step 1: noiseless link with the true channel
    rng = np.random.default_rng(0)
    ch = otfs.random_channels(200, rng, otfs.kmax_from_speed(1000))
    S = otfs.simulate(200, rng, ch, snr_db=300, ep_es_db=14)
    # Step 2: detection must be error-free
    idx = otfs.mrc_detect(S["Yt"], S["g"])
    assert otfs.bit_errors(idx, S["idx"]) == 0


def test_guard_rows_keep_data_out_of_pilot_region():
    # Data-only frame (no pilot, no noise): pilot rows must stay empty
    rng = np.random.default_rng(1)
    ch = otfs.random_channels(20, rng, otfs.kmax_from_speed(1000))
    S = otfs.simulate(20, rng, ch, snr_db=300, ep_es_db=-300)
    assert np.abs(S["Yt"][:, otfs.LP:]).max() < 1e-6


def test_dnn_sizes_match_paper():
    # 2 (lm + 1) N = 80 inputs and 2 (1 + 3P) = 14 outputs
    rng = np.random.default_rng(2)
    ch = otfs.random_channels(5, rng, 1.0)
    S = otfs.simulate(5, rng, ch, 10, 14)
    assert otfs.dnn_input(S["Y"], S["xp"]).shape == (5, 80)
    assert otfs.encode_label(ch, otfs.phase_ref(S["Y"], S["xp"])).shape == (5, 14)


def test_doppler_at_max_speed():
    # 1000 km/h at 3 GHz -> 2778 Hz -> 1.48 Doppler bins
    assert abs(otfs.kmax_from_speed(1000) - 1.4815) < 1e-3
