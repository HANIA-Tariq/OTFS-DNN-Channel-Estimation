"""ZP-OTFS transmitter, channel and receiver (rectangular pulses), batched over B frames."""
import numpy as np
from .config import M, N, MN, L, MD, LP, KP, QAM
from .channel import channel_matrix


def make_frames(B, rng, xp):
    """Delay-Doppler grid of Fig. 2: 4-QAM data in rows 0..MD-1, one pilot at (LP, KP)."""
    idx = rng.integers(0, 4, (B, MD, N))       # random symbol indices
    X = np.zeros((B, M, N), complex)
    X[:, :MD, :] = QAM[idx]                    # data rows
    X[:, LP, KP] = xp                          # pilot; all other guard cells stay 0
    return X, idx


def dd_to_time(X):
    """Delay-Doppler grid -> transmitted time samples s[m + nM]."""
    xt = np.fft.ifft(X, axis=2) * np.sqrt(N)                  # N-point IFFT per delay row
    return xt.transpose(0, 2, 1).reshape(X.shape[0], MN)      # read out column by column


def apply_channel(s, g, sigma2, rng):
    """r[q] = sum_l g[l, q] s[q - l] + noise (paper eq. 5)."""
    r = np.zeros_like(s)
    for tap in range(L):
        shifted = np.zeros_like(s)
        shifted[:, tap:] = s[:, :MN - tap]                    # delay by `tap` samples
        r += g[:, tap, :] * shifted                           # time-varying tap gain
    noise = (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape)) * np.sqrt(sigma2 / 2)
    return r + noise


def time_to_dt(r):
    """Received time samples -> delay-time grid (B, M, N)."""
    return r.reshape(r.shape[0], N, M).transpose(0, 2, 1)


def dt_to_dd(Yt):
    """Delay-time grid -> delay-Doppler grid (N-point FFT per delay row)."""
    return np.fft.fft(Yt, axis=2) / np.sqrt(N)


def simulate(B, rng, ch, snr_db, ep_es_db):
    """Full link for B frames: build frames, pass through channel, receive."""
    sigma2 = 10 ** (-snr_db / 10)              # noise power (data symbol energy = 1)
    xp = np.sqrt(10 ** (ep_es_db / 10))        # pilot amplitude from Ep/Es
    X, idx = make_frames(B, rng, xp)
    g = channel_matrix(ch["h"], ch["k"], ch["l"])
    r = apply_channel(dd_to_time(X), g, sigma2, rng)
    Yt = time_to_dt(r)
    return dict(Y=dt_to_dd(Yt), Yt=Yt, X=X, idx=idx, g=g, xp=xp, sigma2=sigma2)
