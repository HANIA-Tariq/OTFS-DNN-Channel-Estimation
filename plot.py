"""Plot Figs. 4-7 from results/results.json into figures/.

Usage:  python plot.py
"""
import json
import matplotlib
matplotlib.use("Agg")                     # save to file, no window
import matplotlib.pyplot as plt

STYLE = {"perfect": dict(marker="D", ls="-", color="k"),
         "conventional": dict(marker="o", ls="--"),
         "DNN+LS": dict(marker="s", ls=":"),            # checked before "DNN"
         "DNN": dict(marker="^", ls="-")}


def style(name):
    """Line style by method."""
    return next((v for k, v in STYLE.items() if k in name), {})


def plot_figure(results, prefix, key, title, ylabel, fname):
    """Draw every curve whose name starts with `prefix`."""
    plt.figure(figsize=(6, 4.5))
    for name, r in results.items():
        if name.startswith(prefix) and r[key][0] is not None:
            plt.semilogy(r["snr"], r[key], label=name[len(prefix) + 1:], **style(name))
    plt.grid(True, which="both", alpha=0.4)
    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=150)
    plt.close()


def main():
    # Step 1: load results; sort so perfect CSI, conventional, then DNN appear in that order
    results = json.load(open("results/results.json"))
    results = dict(sorted(results.items(), key=lambda kv: ("perfect" not in kv[0], "DNN" in kv[0], kv[0])))

    # Step 2: one figure per paper figure
    plot_figure(results, "fig4", "nmse", "Fig. 4: NMSE, two paths", "NMSE", "fig4_nmse.png")
    plot_figure(results, "fig5", "ber", "Fig. 5: BER, two paths", "BER", "fig5_ber.png")
    plot_figure(results, "fig6", "ber", "Fig. 6: BER at different speeds", "BER", "fig6_ber_speed.png")
    plot_figure(results, "fig7", "ber", "Fig. 7: BER, one path", "BER", "fig7_ber_one_path.png")
    print("figures saved to figures/")


if __name__ == "__main__":
    main()
