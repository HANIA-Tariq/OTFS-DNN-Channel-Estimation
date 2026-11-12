"""Delay-time MRC detector (Thaj & Viterbo, IEEE TVT 2020, Algorithm 2) and bit counting."""
import numpy as np
from .config import M, N, L, LM, MD


def _hard_decision(ct):
    """Delay-time vector -> nearest 4-QAM symbols -> back to delay-time."""
    c = np.fft.fft(ct, axis=-1) / np.sqrt(N)
    d = (np.sign(c.real) + 1j * np.sign(c.imag)) / np.sqrt(2)
    return np.fft.ifft(d, axis=-1) * np.sqrt(N)


def mrc_detect(Yt, g_hat, n_iter=15, n_soft=5):
    """Detect data rows with an estimated channel. Returns QAM indices (B, MD, N)."""
    B = Yt.shape[0]

    # Step 1: channel per tap and delay row, nu[b, l, m, n]
    nu = g_hat.reshape(B, L, N, M).transpose(0, 1, 3, 2)

    # Step 2: MRC normaliser d_m = sum_l |nu_{m+l, l}|^2
    d = np.zeros((B, MD, N))
    for m in range(MD):
        for l in range(L):
            d[:, m] += np.abs(nu[:, l, m + l]) ** 2
    d = np.maximum(d, 1e-12)

    # Step 3: iterate; dy holds the residual (received - reconstructed)
    dy = Yt.copy()
    x = np.zeros((B, MD, N), complex)
    best_x, best_err = x.copy(), np.full(B, np.inf)
    for it in range(n_iter):
        for m in range(MD):
            dg = np.zeros((B, N), complex)
            for l in range(L):
                dg += np.conj(nu[:, l, m + l]) * dy[:, m + l]           # combine all delay branches
            c = x[:, m] + dg / d[:, m]                                  # MRC update
            new = c if it < n_soft else _hard_decision(c)               # soft first, then hard
            delta = new - x[:, m]
            for l in range(L):
                dy[:, m + l] -= nu[:, l, m + l] * delta                 # update residual (DFE)
            x[:, m] = new
        err = np.sum(np.abs(dy[:, :MD + LM]) ** 2, axis=(1, 2))        # residual energy
        better = (err < best_err) | (it == n_soft)
        best_x[better], best_err[better] = x[better], err[better]       # keep best iterate

    # Step 4: back to delay-Doppler and map to QAM index
    c = np.fft.fft(best_x, axis=-1) / np.sqrt(N)
    return (c.real < 0) * 2 + (c.imag < 0) * 1


def bit_errors(idx_hat, idx):
    """Bit errors for Gray-mapped 4-QAM (bit 0 = sign of Im, bit 1 = sign of Re)."""
    e = idx_hat ^ idx
    return np.sum(e & 1) + np.sum((e >> 1) & 1)
