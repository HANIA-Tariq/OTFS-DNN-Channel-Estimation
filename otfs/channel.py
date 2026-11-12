"""Two-path channel with fractional Doppler.

Calibrated to the paper's figures:
- path gains are Rayleigh and normalised to unit total power in every frame, so the
  one-path case (gain of path 2 = 0) gives |h1| = 1, which matches the paper's Fig. 7
  perfect-CSI curve (4-QAM in AWGN) exactly;
- Doppler of each path is uniform in [-k_max, k_max].
"""
import numpy as np
from .config import MN, L, LM, DF, N, FC, C


def kmax_from_speed(v_kmh):
    """Max Doppler shift in units of the Doppler resolution DF / N."""
    nu_max = (v_kmh / 3.6) * FC / C          # max Doppler shift [Hz]
    return nu_max / (DF / N)                 # 1000 km/h -> 1.48 (fractional)


def random_channels(B, rng, kmax, p_single=0.0):
    """Draw B channels. Returns gains h, Doppler k, delays l (each B x 2) and path count P (B,)."""
    # Step 1: Rayleigh gains, normalised to |h1|^2 + |h2|^2 = 1 per frame
    h = (rng.standard_normal((B, 2)) + 1j * rng.standard_normal((B, 2))) / np.sqrt(2)

    # Step 2: optionally remove path 2 (single-path frames)
    P = np.full(B, 2)
    single = rng.random(B) < p_single
    h[single, 1] = 0
    P[single] = 1
    h /= np.linalg.norm(h, axis=1, keepdims=True)

    # Step 3: path 1 at delay 0, path 2 at a random integer delay 1..LM
    l = np.stack([np.zeros(B, int), rng.integers(1, LM + 1, B)], 1)

    # Step 4: fractional Doppler, uniform in [-kmax, kmax]
    k = rng.uniform(-kmax, kmax, (B, 2))
    k[single, 1] = 0
    l[single, 1] = 0
    return dict(h=h, k=k, l=l, P=P)


def channel_matrix(h, k, l):
    """Time-domain channel g[l, q] (paper eq. 6), shape (B, L, MN)."""
    B = h.shape[0]
    q = np.arange(MN)
    g = np.zeros((B, L, MN), complex)
    for i in range(h.shape[1]):                                         # loop over paths
        phase = np.exp(2j * np.pi * k[:, i:i + 1] * (q[None] - l[:, i:i + 1]) / MN)
        g[np.arange(B), l[:, i], :] += h[:, i:i + 1] * phase            # put path on its delay tap
    return g
