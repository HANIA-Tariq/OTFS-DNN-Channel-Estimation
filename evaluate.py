"""Run the NMSE and BER simulations of Figs. 4-7 and save results/results.json.

Usage:  python evaluate.py           (full run, ~40 min on 2 CPU cores)
        python evaluate.py --fast    (2000 frames per point, ~5 min, noisier at low BER)
"""
import argparse
import json
import time
import joblib
import numpy as np
import otfs

SNRS = [8, 10, 12, 14, 16]            # Figs. 4-6 x-axis [dB]
SNRS_FIG7 = [6, 7, 8, 9, 10, 11, 12]  # Fig. 7 x-axis [dB]
BATCH = 2000                          # frames simulated at once


def estimate(method, S, nets):
    """Return the estimated channel matrix g_hat for 'perfect', 'conv', 'dnn' or 'dnn_ls'."""
    if method == "perfect":
        return S["g"]
    if method == "conv":
        return otfs.conventional_estimate(S["Yt"], S["xp"], S["sigma2"])
    ep = int(round(10 * np.log10(S["xp"] ** 2)))                         # pick network by pilot power
    out = nets[ep].predict(otfs.dnn_input(S["Y"], S["xp"]))
    est = otfs.decode_output(out, otfs.phase_ref(S["Y"], S["xp"]))
    if method == "dnn_ls":
        est["h"] = otfs.ls_gains(S["Y"], S["xp"], est)                   # not in the paper: LS gains
    return otfs.channel_matrix(est["h"], est["k"], est["l"])             # rebuild g from parameters


def run_curve(method, ep, nets, fast, snrs=SNRS, speed=otfs.MAX_SPEED_KMH, p_single=0.0, ber=True):
    """One curve: NMSE (and BER) at each SNR."""
    rng = np.random.default_rng(0)                                      # same seed for every curve
    kmax = otfs.kmax_from_speed(speed)
    out = {"snr": snrs, "nmse": [], "ber": []}
    for snr in snrs:
        n_frames = BATCH if fast else (4000 if snr <= 10 else 10_000)   # more frames at low BER
        err = bits = 0
        num = den = 0.0
        for _ in range(n_frames // BATCH):
            ch = otfs.random_channels(BATCH, rng, kmax, p_single=p_single)
            S = otfs.simulate(BATCH, rng, ch, snr, ep)
            g_hat = estimate(method, S, nets)
            num += np.sum(np.abs(S["g"] - g_hat) ** 2)                  # NMSE numerator
            den += np.sum(np.abs(S["g"]) ** 2)                          # NMSE denominator
            if ber:
                idx = otfs.mrc_detect(S["Yt"], g_hat)
                err += otfs.bit_errors(idx, S["idx"])
                bits += idx.size * 2                                    # 2 bits per 4-QAM symbol
        out["nmse"].append(float(num / den))
        out["ber"].append(float(err / bits) if ber else None)
    return out


def all_curves():
    """Every curve of Figs. 4-7 -> (name, method, Ep/Es, options)."""
    curves = []
    curves += [(f"fig4 conventional Ep/Es={ep}", "conv", ep, dict(ber=False)) for ep in (11, 20, 30)]
    curves += [(f"fig4 DNN Ep/Es={ep}", "dnn", ep, dict(ber=False)) for ep in (14, 23)]
    curves += [(f"fig4 DNN+LS Ep/Es={ep}", "dnn_ls", ep, dict(ber=False)) for ep in (14, 23)]
    curves += [("fig5 perfect CSI", "perfect", 20, {})]
    curves += [(f"fig5 conventional Ep/Es={ep}", "conv", ep, {}) for ep in (20, 30, 40)]
    curves += [(f"fig5 DNN Ep/Es={ep}", "dnn", ep, {}) for ep in (14, 23)]
    curves += [(f"fig5 DNN+LS Ep/Es={ep}", "dnn_ls", ep, {}) for ep in (14, 23)]
    curves += [(f"fig6 DNN Ep/Es={ep}, 500 km/h", "dnn", ep, dict(speed=500)) for ep in (14, 23)]
    one_path = dict(p_single=1.0, snrs=SNRS_FIG7)                       # Fig. 7: path 2 removed
    curves += [("fig7 perfect CSI", "perfect", 20, one_path)]
    curves += [(f"fig7 conventional Ep/Es={ep}", "conv", ep, one_path) for ep in (20, 40)]
    curves += [(f"fig7 DNN Ep/Es={ep}", "dnn", ep, one_path) for ep in (14, 23)]
    curves += [(f"fig7 DNN+LS Ep/Es={ep}", "dnn_ls", ep, one_path) for ep in (14, 23)]
    return curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fast", action="store_true", help="fewer frames per SNR point")
    args = p.parse_args()

    # Step 1: load the trained networks
    nets = {ep: joblib.load(f"models/dnn_ep{ep}.joblib")["model"] for ep in (14, 23)}

    # Step 2: list all curves
    curves = all_curves()

    # Step 3: simulate every curve and save after each one
    results, t0 = {}, time.time()
    for name, method, ep, opts in curves:
        results[name] = run_curve(method, ep, nets, args.fast, **opts)
        print(f"{name:32s} BER {results[name]['ber']}", flush=True)
        json.dump(results, open("results/results.json", "w"), indent=1)

    # Step 4: Fig. 6 reuses the 1000 km/h curves of Fig. 5
    results["fig6 perfect CSI"] = results["fig5 perfect CSI"]
    for ep in (14, 23):
        results[f"fig6 DNN Ep/Es={ep}, 1000 km/h"] = results[f"fig5 DNN Ep/Es={ep}"]
    json.dump(results, open("results/results.json", "w"), indent=1)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
