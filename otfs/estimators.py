"""Channel estimators: DNN input/output encoding and the conventional pilot-based method."""
import numpy as np
from scipy.interpolate import CubicSpline
from .config import M, N, MN, L, LM, LP, KP


# ---------------------------------------------------------------------------
# DNN estimator: received pilot -> channel parameters {h, k, l, P}
# ---------------------------------------------------------------------------
def phase_ref(Y, xp):
    """Per-delay-row phase reference: rotates each pilot row so its strongest sample is real."""
    Yp = Y[:, LP:, :]                                                   # pilot rows (B, L, N)
    peak = np.argmax(np.abs(Yp), axis=2)[..., None]                     # strongest sample per row
    ref = np.take_along_axis(Yp, peak, axis=2)[..., 0]
    return np.conj(ref) / np.maximum(np.abs(ref), 1e-12)                # unit phasor (B, L)


def dnn_input(Y, xp):
    """80 real DNN inputs: Re/Im of pilot rows LP..M-1, scaled by pilot and phase-aligned."""
    Yp = Y[:, LP:, :] / xp * phase_ref(Y, xp)[..., None]
    Yp = Yp.reshape(Y.shape[0], -1)                                     # (B, 40) complex
    return np.concatenate([Yp.real, Yp.imag], axis=1)                   # (B, 80) real


def encode_label(ch, rot):
    """14 real targets: Re/Im of {h1, h2, k1, k2, l1, l2, P}."""
    h = ch["h"] * np.take_along_axis(rot, ch["l"], axis=1)              # gains in the aligned phase
    z = np.concatenate([h, ch["k"] + 0j, ch["l"] + 0j, ch["P"][:, None] + 0j], axis=1)
    return np.concatenate([z.real, z.imag], axis=1)


def decode_output(out, rot):
    """DNN output (B, 14) -> channel parameters."""
    l = np.clip(np.rint(out[:, 4:6]), 0, LM).astype(int)                # round delays to taps
    h = (out[:, 0:2] + 1j * out[:, 7:9]) * np.conj(np.take_along_axis(rot, l, axis=1))   # undo phase
    k = out[:, 2:4]                                                     # fractional Doppler
    P = np.clip(np.rint(out[:, 6]), 1, 2).astype(int)                   # number of paths
    h = h.copy()
    h[P == 1, 1] = 0                                                    # single path: drop path 2
    return dict(h=h, k=k, l=l, P=P)


# ---------------------------------------------------------------------------
# Optional (not in the paper): least-squares refinement of the path gains
# ---------------------------------------------------------------------------
def pilot_response(k, l, xp):
    """Noiseless received pilot rows (B, 40) for one unit-gain path with Doppler k and delay l."""
    from .system import dd_to_time, time_to_dt, dt_to_dd                # avoid circular import
    from .channel import channel_matrix
    B = len(k)
    X = np.zeros((B, M, N), complex)
    X[:, LP, KP] = xp                                                   # pilot only
    g = channel_matrix(np.ones((B, 1)), k[:, None], l[:, None])
    s = dd_to_time(X)
    r = np.zeros_like(s)
    for tap in range(L):
        shifted = np.zeros_like(s)
        shifted[:, tap:] = s[:, :MN - tap]
        r += g[:, tap] * shifted
    return dt_to_dd(time_to_dt(r))[:, LP:, :].reshape(B, -1)


def ls_gains(Y, xp, est):
    """Re-estimate the gains by least squares, keeping the DNN's delays, Dopplers and P."""
    yp = Y[:, LP:, :].reshape(Y.shape[0], -1)                           # received pilot rows (B, 40)
    A = np.stack([pilot_response(est["k"][:, i], est["l"][:, i], xp) for i in range(2)], 2)   # (B, 40, 2)
    A[est["P"] == 1, :, 1] = 0                                          # single path: one column
    AH = A.conj().transpose(0, 2, 1)
    R = AH @ A + 1e-9 * np.eye(2)                                       # 2x2 normal equations
    h = np.linalg.solve(R, (AH @ yp[..., None]))[..., 0]               # h = (A^H A)^-1 A^H y
    h[est["P"] == 1, 1] = 0
    return h


# ---------------------------------------------------------------------------
# Conventional estimator: delay-time domain pilot + cubic-spline interpolation
# ---------------------------------------------------------------------------
def conventional_estimate(Yt, xp, sigma2=None):
    """Estimate g[l, q] of all LM+1 taps at the N pilot instants, then interpolate over the frame."""
    n = np.arange(N)
    q = np.arange(MN)
    xt_p = xp * np.exp(2j * np.pi * KP * n / N) / np.sqrt(N)            # pilot in delay-time domain
    g_hat = np.zeros((Yt.shape[0], L, MN), complex)
    for tap in range(L):
        est = Yt[:, LP + tap, :] / xt_p[None]                           # gain at q = LP + tap + nM
        qs = LP + tap + n * M                                           # sample positions of estimates
        g_hat[:, tap] = CubicSpline(qs, est, axis=1, bc_type="natural")(q)   # interpolate whole frame
    return g_hat


def nmse(g, g_hat):
    """Normalised mean squared error of the channel matrix."""
    return np.sum(np.abs(g - g_hat) ** 2) / np.sum(np.abs(g) ** 2)
